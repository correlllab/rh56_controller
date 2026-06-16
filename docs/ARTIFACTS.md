# Optional Artifacts

The main repository should stay small enough for users who only need the code,
install profiles, MuJoCo assets, and README demo previews. Large experiment
logs, raw video sources, and one-off analysis outputs should live outside the
default checkout.

## Kept In This Repository

- Curated README demos in `resource/video/`:
  - `force_control.mp4`
  - `force_control_gif_source.gif`
  - `nut1.mp4`
  - `nut1.gif`
  - `nut2.mp4`
  - `nut2.gif`
  - `cube.mp4`
  - `cube.gif`
  - `2finger_pinch_long_obj.mp4`
  - `2finger_pinch_long_obj.gif`
  - `4finger_grab_long_obj.mp4`
  - `4finger_grab_long_obj.gif`
  - `3d_workspace.png`
- Small benchmark summaries and plots in `resource/experiment/`.
- Scripts needed to regenerate or inspect results.

## Kept Out Of The Default Checkout

- `experiment_data/`
- Raw or source video files such as `resource/video/*.mov`.
- Extra exported demos that are not linked from the README.
- One-off CSV logs such as `middlefinger_speed_sweep_*.csv`.
- Generated plots, papers, and local logs.

## Suggested Local Layout

For lab machines, keep the optional data beside the repo:

```text
workspace/
  rh56_controller/
  rh56_artifacts/
    experiment_data/
    resource/
      video/
        raw/
```

If the artifacts need to be shared through GitHub, use a separate artifacts
repository and clone it only when needed. A private or public artifact repo can
also be added as an optional submodule later, but normal users should not need
it for installation or simulation.

## Adding New Demo Media

Only add media to `resource/video/` when it is intentionally used by the README
or documentation. Because `resource/video/` is allowlisted in `.gitignore`, new
demo files may need to be added explicitly with `git add -f`.
