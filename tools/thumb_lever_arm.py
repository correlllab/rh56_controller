#!/usr/bin/env python3
"""
Compute the effective lever arm r_eff(theta_flex) for the thumb yaw motor.

The moment arm is the perpendicular distance from the yaw axis to the fingertip,
projected onto the plane normal to the yaw axis.  This varies with thumb flexion
because the tip traces an arc as the thumb bends.

Outputs a 2nd-order polynomial fit r_eff(theta_flex_rad) for use at runtime
in the tangential force recovery.

Usage:
    python tools/thumb_lever_arm.py
    python tools/thumb_lever_arm.py --plot
"""

import sys
import argparse
import numpy as np
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mujoco

_DEFAULT_XML = str(_REPO_ROOT / "h1_mujoco" / "inspire" / "inspire_right.xml")


def compute_lever_arm(xml_path: str = _DEFAULT_XML) -> np.ndarray:
    """Run FK sweep and return polynomial coefficients for r_eff(theta_flex).

    Returns np.ndarray of shape (3,): [c2, c1, c0] for np.polyval(poly, theta).
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    yaw_jnt_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "thumb_proximal_yaw_joint")
    flex_jnt_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "thumb_proximal_pitch_joint")
    tip_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_thumb_tip")
    yaw_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "thumb_proximal_base")

    print(f"yaw_jnt_id={yaw_jnt_id}, flex_jnt_id={flex_jnt_id}")
    print(f"tip_site_id={tip_site_id}, yaw_body_id={yaw_body_id}")

    flex_angles = np.linspace(0, 0.6, 30)
    r_effs = []

    for theta_flex in flex_angles:
        data.qpos[:] = 0
        # Set yaw to neutral (0 = spread out in sim)
        data.qpos[model.jnt_qposadr[yaw_jnt_id]] = 0.0
        # Set flex angle
        data.qpos[model.jnt_qposadr[flex_jnt_id]] = theta_flex
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        p_tip = data.site_xpos[tip_site_id].copy()
        p_yaw = data.xpos[yaw_body_id].copy()
        R_base = data.xmat[yaw_body_id].reshape(3, 3)
        z_yaw = R_base @ np.array([0.0, 0.0, -1.0])

        r_vec = p_tip - p_yaw
        r_axial = np.dot(r_vec, z_yaw) * z_yaw
        r_perp = r_vec - r_axial
        r_eff = np.linalg.norm(r_perp)

        r_effs.append(r_eff)
        print(f"  flex={np.degrees(theta_flex):5.1f}°  r_eff={r_eff*1000:6.2f} mm")

    r_effs = np.array(r_effs)

    # Fit 2nd-order polynomial
    poly = np.polyfit(flex_angles, r_effs, 2)
    print(f"\n2nd-order polynomial fit: r_eff = {poly[0]:.6f}*theta^2 + {poly[1]:.6f}*theta + {poly[2]:.6f}")
    print(f"numpy.polyval coefficients (c2, c1, c0): {poly.tolist()}")

    # Validate fit
    residuals = r_effs - np.polyval(poly, flex_angles)
    print(f"Fit residuals: max={np.abs(residuals).max()*1000:.3f} mm, "
          f"RMS={np.sqrt(np.mean(residuals**2))*1000:.3f} mm")

    return poly, flex_angles, r_effs


def main():
    parser = argparse.ArgumentParser(description="Compute thumb yaw lever arm polynomial")
    parser.add_argument("--xml", type=str, default=_DEFAULT_XML)
    parser.add_argument("--plot", action="store_true", help="Show plot of r_eff vs flex angle")
    args = parser.parse_args()

    poly, flex_angles, r_effs = compute_lever_arm(args.xml)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            theta_dense = np.linspace(0, 0.6, 200)
            r_fit = np.polyval(poly, theta_dense)
            plt.figure(figsize=(8, 5))
            plt.plot(np.degrees(flex_angles), r_effs * 1000, "o", label="MuJoCo FK")
            plt.plot(np.degrees(theta_dense), r_fit * 1000, "-", label="2nd-order poly fit")
            plt.xlabel("Flex angle θ (degrees)")
            plt.ylabel("r_eff (mm)")
            plt.title("Thumb yaw lever arm vs. flexion angle")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
