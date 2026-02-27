# Grasp Geometry Planning & Visualization — Inspire RH56

## Overview

`grasp_geometry.py` + `grasp_viz.py` form an offline kinematic grasp planner for the
Inspire RH56 dexterous hand. It computes the hand base pose and joint angles needed to
execute antipodal grasps at a specified width/diameter and world-frame height, using
three grasp primitives: **line** (2-finger pinch), **plane** (3–5 finger precision), and
**cylinder** (power grasp).

The central problem is the **coupling offset**: as fingers flex, fingertips shift ~75 mm in
the hand's Z axis (extension direction). The planner solves for the base pose that
compensates for this, so that the grasp midplane stays at the target world height.

---

## Files

| File | Purpose |
|---|---|
| `rh56_controller/grasp_geometry.py` | FK model + closure solver |
| `rh56_controller/grasp_viz.py` | Interactive matplotlib + MuJoCo visualizer |
| `h1_mujoco/inspire/inspire_right.xml` | Ground-truth FK model (read-only) |
| `h1_mujoco/inspire/inspire_grasp_scene.xml` | Floating hand for 3D viewer |
| `h1_mujoco/inspire/.fk_cache.npz` | Cached FK sweep tables (auto-rebuilt) |

---

## Joint Coupling (from `inspire_right.xml` equality constraints)

The XML `polycoef="a0 a1"` convention defines `joint1 = a0 + a1 * joint2`.

| Joint pair | Equation |
|---|---|
| pinky/ring/middle intermediate = proximal | `intermediate = proximal` (1:1) |
| index intermediate | `intermediate = 0.15 + 1.0 × proximal` |
| thumb intermediate | `intermediate = 0.15 + 1.25 × pitch` |
| thumb distal | `distal = 0.15 + 0.75 × pitch` |

The 0.15 rad preload means all coupled joints are bent ~8.6° even at zero proximal angle.

---

## Coordinate Conventions

### Hand base frame
- **+Z**: finger extension direction (wrist → fingertip when fully open)
- **+Y**: finger spread (index at Y ≈ +32 mm, pinky at Y ≈ −26 mm)
- **+X**: palm closure direction (fingertips curl from +Z toward +X when flexed)

### World frame (top-down approach)
The combined rotation **R = Ry(θ) ∘ Rx(π)** maps base → world:
- Rx(π): base +Z → world −Z (fingers hang down), base +Y → world −Y
- Ry(θ): additional tilt about world Y to equalize finger and thumb heights

Explicit matrix:
```
R = [[ cos θ,  0, −sin θ ],
     [   0,   −1,    0   ],
     [−sin θ,  0, −cos θ ]]
```

---

## FK Model (`InspireHandFK`)

A MuJoCo FK sweep precomputes tip positions as functions of joint angles, then fits
scipy interpolators for fast lookup at runtime.

### Sweep procedure
For each non-thumb finger f with `ctrl ∈ [0, ctrl_max[f]]` (200 samples):
1. Zero all qpos
2. Set `qpos[proximal] = ctrl`, `qpos[intermediate]` = coupled value from table above
3. Call `mj_kinematics` (not `mj_step` — no dynamics)
4. Record `site_xpos[tip_site]` in base frame

For the thumb (50 × 50 grid over pitch × yaw):
1. Set `qpos[yaw]`, `qpos[pitch]`, and both coupled joints
2. Record tip position

### Coupling pitfall
The FK sweep must set `data.qpos` directly, **not** `data.ctrl`. MuJoCo equality
constraints are not enforced by `mj_kinematics` alone — the caller must compute the
coupled joint values explicitly using the polycoef equations above.

---

## Closure Geometry (`ClosureGeometry`)

### Tilt derivation

The tilt angle θ is the rotation about world Y that makes the grasp plane horizontal
(thumb tip and finger centroid at the same world Z).

Let **d = T − C** where T = thumb tip, C = finger tip (or centroid). Requiring
`(R d)_z = 0`:

```
−sin(θ) d_x − cos(θ) d_z = 0
→  θ = atan2(−d_z, d_x)
```

### Effective grasp width identity

The world-frame X separation equals:
```
(R d)_x = cos(θ) d_x − sin(θ) d_z = d_x²/r + d_z²/r = √(d_x² + d_z²)
```
where r = hypot(d_x, d_z) and θ is the tilt that zeroes (R d)_z.

**Consequence**: the graspable width is the XZ-plane distance between thumb and finger
tips, regardless of their Y (anatomical spread) or Z difference. Using 3D distance
includes the Y offset, which is not closeable by rotation.

### Tilt modal shift

`atan2(−d_z, d_x)` is in (−π, π). When `d_x > 0` (thumb X > finger X, normal
opposition), the result is already in (−π/2, π/2). When `d_x ≤ 0` (thumb has moved past
the finger in X at maximum closure — the "narrow regime"), the angle exceeds ±π/2.

**Wrong fix**: wrapping by ± π creates a ~155° discontinuity in hand pose as the slider
moves through s_d0 (the closure parameter where `d_x = 0`).

**Current approach — clip**:
```
θ_applied = clip(atan2(−d_z, d_x),  −π/2,  π/2)
```

For `d_x ≤ 0`, the hand holds a fixed 90° side-approach orientation (fingers pointing
sideways in world frame).  The residual coplanarity error is `|d_x| ≲ 5 mm`, which is
acceptable for practical grasps.  This is monotone and smooth by construction — tilt
increases toward 90° as width shrinks to `width_d0 ≈ 14.1 mm` (index), then stays
frozen at 90° through the remaining 8.7–14.1 mm range.

**Key closed-form quantities (index reference finger)**:

| Quantity | Value |
|---|---|
| s_d0 (d_x = 0 on proportional trajectory) | 0.437 |
| width at s_d0 | 14.1 mm |
| s_min (minimum XZ distance) | 0.487 |
| minimum achievable width | 8.7 mm |

### Closure parameter `s`

For line and plane modes, both thumb pitch and reference-finger ctrl scale with a single
parameter s ∈ [0, 1]:
- `ctrl_pitch = s × ctrl_max[thumb_proximal]`
- `ctrl_ref = s × ctrl_max[ref_finger]`

The XZ distance `D(s) = hypot(T(s) − I(s))` is monotone-decreasing on [0, s_min].
`brentq` finds the s where D(s) = target_width.

### Coplanarity correction

After the joint closure solve, the reference finger's base-frame Z (`target_z`) is used
to bring all other active fingers to the same height via `brentq` on `z(ctrl) − target_z`.

Direction-aware fallback when `target_z` is out of range for a finger:
- `target_z ≥ z_max` → `ctrl = 0` (fully extended — highest achievable Z)
- `target_z < z_min` → `ctrl = ctrl_max` (fully curled — lowest achievable Z)

### Cylinder mode

Power grasps use the **proximal-intermediate joint positions** (end of proximal links)
as the diameter model, not fingertips. The object sits between the proximal links.

```
diameter = ||T_prox − mean(finger_prox)||
```

Thumb yaw: above the pre-computed transition diameter (71.6 mm), thumb opposes at
`yaw = 1.308` rad; below, palm mode uses `yaw = 0` and `tilt = 0`.

---

## World Frame Conversion

Given a `ClosureResult` (all positions in base frame):

```python
R = Ry(tilt_y) @ Rx(π)
mid_w = R @ midpoint
base_w = [−mid_w[0], −mid_w[1], gz − mid_w[2]]   # base origin in world
tip_w  = R @ tip + base_w                           # each tip in world
```

The hand base is always above `gz` (base_w[2] ≥ gz) because `mid_w[2] ≤ 0` for all
valid configurations (the midpoint's world Z contribution is non-positive).

---

## MuJoCo Viewer Sync

The floating hand model (`inspire_grasp_scene.xml`) has 12 DOF: 6 base (pos_x/y/z,
rot_x/y/z) + 6 finger joints. `_apply_qpos` writes directly to `data.qpos` and calls
`mj_kinematics` (bypassing the MuJoCo equality-constraint solver; coupled joints are
computed explicitly using the polycoef equations).

Viewer rotation convention: MuJoCo applies `Rx(rot_x) ∘ Ry(rot_y)` (axes in parent
frame). To match FK's `R = Ry(tilt) ∘ Rx(π)`, use:

```
rot_x = π,  rot_y = −tilt_y
```

Identity: `Rx(π) ∘ Ry(−θ) = Ry(θ) ∘ Rx(π)`.

---

## Verified Width Ranges

| Mode | Min (mm) | Max (mm) | Notes |
|---|---|---|---|
| 2-finger line | 8.7 | 122 | Ramp region 8.7–14.1 mm; tilt passes through 0° at ≈9.3 mm |
| 3/4/5-finger plane | 15.6 | 130 | Ramp region 15.6–24.9 mm |
| cylinder | 28.6 | 103 | Palm mode (tilt=0°) below 71.6 mm diameter |

---

## Attempted Improvements and Why They Failed

The narrow-width regime (below ~14 mm for line, ~25 mm for plane) is where the
proportional s-trajectory produces large tilt (approaching 90°) because the thumb tip
crosses the finger tip in X (`d_x → 0 → negative`). Two alternative strategies were
attempted and both caused severe regressions.

### Attempt 1 — Blend ramp (`_compute_blend_params`)

**Idea**: Precompute `s_d0` (where `d_x = 0`) and `s_min` (XZ minimum). For
`s ∈ [s_d0, s_min]`, linearly blend tilt from `π/2` toward `tilt_target ≈ −35°`,
a "pitch back up" effect. Ctrl values remain proportional (smooth by construction).

**Failure mode**: The linear tilt ramp passes through 0° and continues to −35° (backward
tilt), causing the hand to visibly flip orientation as the slider moves through the
8.7–14.1 mm range. The tilt excursion (90° → −35° in ~5 mm of width change) was too
aggressive and looked jarring even though the ctrl values were smooth.

**Root cause**: The "exact coplanarity" tilt in the narrow regime lives on the upper
branch `∈ (90°, 148°)` — physically inaccessible for a top-down grasp. Any smooth
approximation must traverse a large angular distance, and there is no way to do so
without a visually jarring rotation.

---

### Attempt 2 — 2D minimum-tilt search

**Idea**: At each target width, sweep `ctrl_pitch` over 50 values; for each find the
`ctrl_ref` (via `brentq`) that achieves the target width; pick the pair that minimises
`|atan2(−d_z, d_x)|`. Ctrl values are no longer constrained to the proportional line.

**Failure mode**: Hand jitters severely as the width slider moves — large discontinuous
jumps in both `ctrl_pitch` and `ctrl_ref` at adjacent widths.

**Root cause** (user's diagnosis): The 2D search back-actuates the thumb at **every**
width independently, finding configurations with wildly different ctrl values at adjacent
widths. The argmin of the tilt landscape is a shallow, non-convex function of `(ctrl_pitch,
ctrl_ref)` and the global minimiser jumps discontinuously. Smoothness was sacrificed
entirely.

---

### Attempt 3 — Precomputed coplanar table

**Idea**: Sweep `ctrl_ref` monotonically from 0 to `CTRL_MAX[ref_finger]`. For each
`ctrl_ref`, find the coplanar `ctrl_pitch` via `fk.thumb_tip_at_z(I[2], max_yaw)` —
i.e., force `T[2] = I[2]` (zero tilt). Store as `interp1d` tables; replace the runtime
search with an O(1) lookup.

**Failure mode**: The user reverted before visual evaluation. In retrospect, the coplanar
trajectory replaces the full proportional trajectory for ALL widths (not just narrow
ones), fundamentally changing the hand pose even at wide widths where the original was
already good. Continuity at the transition from wide to narrow was not guaranteed.

---

### Lesson

The fundamental difficulty is the 2-DOF nature of the thumb (`ctrl_pitch`, `ctrl_yaw`)
combined with the kinematic constraint (target width) leaving only one free DOF, and
the tilt objective — which lives in the 2D `(ctrl_pitch, ctrl_ref)` space — having a
non-convex, wide-flat landscape. Any strategy that searches this space independently at
each width will find discontinuous solutions.

The proportional trajectory (`ctrl_pitch = s × 0.6`, `ctrl_ref = s × ctrl_max`) works
because it is a *global* 1D parameterisation through a smooth region of the space. The
small imperfection (90° side-approach at narrow widths) is the price of smoothness, and
it is empirically acceptable.

**If further improvement is attempted**, the only safe approach is to modify the
proportional trajectory as a small, continuous perturbation — e.g., a correction term
that activates only when `d_x < ε` and is itself smooth in `s`. Any global re-optimisation
that searches the 2D ctrl space at each width independently is likely to reproduce the
jitter observed in Attempt 2.

---

## Running

```bash
python -m rh56_controller.grasp_viz            # interactive visualizer
python -m rh56_controller.grasp_geometry       # self-test (rebuild cache if needed)
python -m rh56_controller.grasp_geometry --rebuild   # force FK table rebuild
```
