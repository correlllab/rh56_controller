# Thumb Yaw Tangential Force Estimation

## Problem Statement

The thumb yaw motor (`thumb_proximal_yaw_joint`) is a rotational DOF that sweeps the entire
thumb in/out relative to the palm. When the thumb tip contacts an object and a tangential
(slip-direction) force acts at the contact point, it creates a torque about the yaw axis that
the yaw motor "feels" via back-drive current. `force_act()[5]` measures this as a raw 0–1000
value — a torque, not a force.

Goal: convert `force_act()[5]` (yaw torque proxy) into a tangential contact force vector.

---

## Where the Angle Conversion Lives

Two places to check/flip signs for the thumb yaw:

| Location | Code | Notes |
|---|---|---|
| `rh56_driver.py:156` | `q = (raw / 1000.0) * π` | Linear, no sign flip |
| `mujoco_bridge.py:212` | `real = 1000 - (ctrl / 1.308 * 1000)` | **Inverted** — 0 raw = max yaw, 1000 raw = zero yaw |

So in the real hand: `raw=0` → thumb yawed in (adducted), `raw=1000` → thumb spread out (abducted).
In MuJoCo: `ctrl=0` → spread out, `ctrl=1.308` → yawed in. The sign of force_act()[5] likely
follows the raw convention. **Verify experimentally by pushing thumb tangentially and observing
which direction gives positive force_act()[5].**

---

## Physical Model

```
            ẑ_yaw  (yaw axis, ≈ palm normal)
               |
               |  r_eff(θ_flex)
         [yaw joint] ──────────────► [fingertip]
                      r̂ (radial)
                              ↑
                           F_tangential (causes τ_yaw)
                           direction: ẑ_yaw × r̂
```

The key relationship:

```
τ_yaw = F_tangential_⊥  ×  r_eff(θ_flex)
```

Where:
- `τ_yaw` is the torque about the yaw axis (what the motor measures)
- `F_tangential_⊥` is the component of tangential force perpendicular to both the yaw axis
  and the moment arm (the only component that torques the yaw motor)
- `r_eff(θ_flex)` is the moment arm = perpendicular distance from yaw axis to the fingertip,
  projected onto the plane normal to the yaw axis

**Critical limitation:** The yaw motor only senses ONE component of the 2D tangential force.
A force directed radially (along the `yaw_axis → tip` vector) creates zero yaw torque and is
invisible here. We cannot recover the full x,y tangential force from yaw alone.

---

## Step 1 — Lever Arm vs. Flexion Angle (MuJoCo)

The moment arm `r_eff` varies with thumb bend because the tip traces an arc as the thumb flexes.
Note from the XML: `thumb_intermediate_joint` and `thumb_distal_joint` are coupled 1:1 to
`thumb_proximal_pitch_joint` (the bend/flex joint), so one angle controls all three.

**Script to write:** `tools/thumb_lever_arm.py`

```python
import mujoco, numpy as np

model = mujoco.MjModel.from_xml_path("h1_mujoco/inspire/inspire_right.xml")
data  = mujoco.MjData(model)

yaw_jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thumb_proximal_yaw_joint")
flex_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thumb_proximal_pitch_joint")
tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,  "right_thumb_tip")
yaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "thumb_proximal_base")

flex_angles = np.linspace(0, 0.6, 30)  # thumb_bend ctrl range
results = []

for theta_flex in flex_angles:
    # Lock yaw at neutral (ctrl=0, spread-out in sim), sweep flex
    data.qpos[:] = 0
    data.qpos[flex_jnt_id] = theta_flex        # coupled joints handled by equality
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    p_tip      = data.site_xpos[tip_site_id].copy()          # tip in world frame
    p_yaw      = data.xpos[yaw_body_id].copy()               # yaw joint origin
    # Yaw axis in world frame (joint axis "0 0 -1" in body frame)
    R_base     = data.xmat[yaw_body_id].reshape(3, 3)
    z_yaw_world = R_base @ np.array([0, 0, -1])              # world-frame yaw axis

    # Moment arm: project (tip - yaw_origin) onto plane ⊥ to yaw axis
    r_vec  = p_tip - p_yaw
    r_axial = np.dot(r_vec, z_yaw_world) * z_yaw_world       # component along axis
    r_perp  = r_vec - r_axial                                 # component in ⊥ plane
    r_eff   = np.linalg.norm(r_perp)

    results.append((theta_flex, r_eff))
    print(f"flex={np.degrees(theta_flex):.1f}°  r_eff={r_eff*1000:.1f} mm")
```

Expected output: `r_eff` grows from ~30 mm (open/flat) toward ~55 mm (fully flexed) as the
tip swings away from the axis. Fit with a 2nd-order polynomial for use at runtime.

---

## Step 2 — Torque Calibration

The raw value `force_act()[5]` is a torque proxy, not a force. Two calibration paths:

**Interim (use now):** Borrow the thumb bend mapping as an order-of-magnitude estimate.
The thumb bend calibration is `F_N = 0.012547 × raw − 0.384`. If we treat this as
`τ_yaw [N·m] ≈ (0.012547 × raw − 0.384) × r_nominal` where `r_nominal ≈ 0.045 m`,
we get a rough torque scale of ~0.00056 N·m / raw unit.

**Proper calibration (do later):** Mount thumb against a force gauge, apply known lateral
forces at the fingertip, record `force_act()[5]` vs. known torque. Fit
`τ_yaw [N·m] = a_yaw × raw + b_yaw`.

---

## Step 3 — Force Recovery

Once we have `τ_yaw` in N·m and `r_eff(θ_flex)` in meters:

```python
def thumb_yaw_tangential_force(
    raw_yaw: float,
    theta_flex_raw: int,         # force_act index 4 angle? No — use angle_read()[5]
    calib: tuple = (0.012547, -0.384),
    r_poly: np.ndarray = None,   # polynomial coefficients from Step 1
) -> dict:
    """
    Returns:
      magnitude:  |F_tangential_yaw_sensitive| in Newtons
      direction:  unit vector in world frame (ẑ_yaw × r̂_tip)  — needs current pose
    """
    a, b = calib
    tau_yaw = a * raw_yaw + b                             # N·m (interim calibration)
    theta_flex_rad = (theta_flex_raw / 1000.0) * 0.60    # raw → rad (flex ctrl range)
    r_eff = np.polyval(r_poly, theta_flex_rad)            # from fitted polynomial
    F_magnitude = abs(tau_yaw) / r_eff if r_eff > 1e-3 else 0.0
    # Direction vector requires current kinematics (see Step 4)
    return {"N": F_magnitude, "tau_yaw_Nm": tau_yaw, "r_eff_m": r_eff}
```

**Input angles needed:**
- `angle_read()[4]` = thumb bend raw angle → convert to flex radians: `(raw / 1000) * 0.60`
- `angle_read()[5]` = thumb yaw raw angle → needed for force direction (not magnitude)

---

## Step 4 — Expressing as x, y Components

The recovered force direction in world frame:

```python
# At runtime with current pose:
z_yaw = R_base @ [0, 0, -1]        # world-frame yaw axis (from mujoco or approximated)
r_vec = p_tip - p_yaw_origin       # from forward kinematics
r_perp = r_vec - dot(r_vec, z_yaw)*z_yaw
r_hat = r_perp / norm(r_perp)
F_direction = cross(z_yaw, r_hat)  # unit vector perpendicular to both axis and radius

F_vec = F_magnitude * sign(tau_yaw) * F_direction   # 3D force vector
```

For x, y in the palm frame (simplest, no FK needed):
```python
F_x = dot(F_vec, palm_x_hat)   # spread / ab-adduction direction
F_y = dot(F_vec, palm_y_hat)   # extension / proximal-distal direction
```

**Note on the missing component:** A force along `r̂` (radially toward/away from the yaw axis,
roughly pointing into the palm) creates zero yaw torque. If the primary slip direction is
radial for your grasp geometry, this sensor is blind to it.

---

## Sign Convention Check

Before implementing, run this quick test:
1. Flex thumb partway (~500 raw), hold against a surface
2. Push thumb in the **adduction direction** (toward index finger) with a gentle lateral force
3. Check sign of `force_act()[5]`: should be consistent and nonzero
4. Then push in the **abduction direction** (away from index)
5. Verify sign flips

If sign is backwards from expectation, flip in the conversion: `tau_yaw = -(a * raw + b)`.

---

## Decision Points

| Decision | Options | Recommended |
|---|---|---|
| Lever arm source | (A) MuJoCo polynomial fit, (B) fixed nominal value | A for accuracy, B for quick start |
| Torque calibration | (A) Borrow thumb_bend coeffs, (B) dedicated calibration | A for now, B later |
| Output format | (A) Scalar N, (B) 3D vector, (C) 2D palm-frame x,y | A first, then B/C |
| Sign check | Must run before trusting any numbers | Do this before anything else |
| Missing radial component | Accept limitation, or add palm-normal force proxy | Accept for now |

---

## Implementation Order

1. **Sign check** (5 min) — verify `force_act()[5]` sign convention experimentally
2. **MuJoCo lever arm script** (30 min) — `tools/thumb_lever_arm.py`, output polynomial
3. **`thumb_yaw_tangential_N()`** in `rh56_hand.py` (20 min) — scalar force using interim calib
4. **Integrate into `real2sim_viz.py`** — add yaw tangential force to force closure analysis
5. **Dedicated torque calibration** (1 hr) — proper `a_yaw`, `b_yaw` coefficients
6. **3D/2D force vector output** — add direction via live FK or lookup table

---

## Open Questions

- Does the yaw motor force reading saturate or have a different noise floor than flexion motors?
  (Yaw motor may be less stiff → lower force sensitivity)
- Is the contact primarily at the distal pad (favoring the tip site position) or distributed?
- Should `r_eff` also account for yaw angle changes (the tip moves laterally as yaw changes)?
  At small yaw angles this is second-order; revisit if accuracy matters.
- At `force_act()[5]` ≈ 0 with contact known from flexion motor, does that mean radial-only
  slip, or no slip at all? Worth logging both simultaneously during experiments.
