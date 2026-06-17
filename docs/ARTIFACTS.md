# Optional Artifacts

The main repository should stay small enough for users who only need the code,
install profiles, MuJoCo assets, and written documentation. Large experiment
logs, videos, project media, and one-off analysis outputs should live outside
the default checkout.

## Kept In This Repository

- Small benchmark summaries and plots in `resource/experiment/`.
- Scripts needed to regenerate or inspect results.

## Kept Out Of The Default Checkout

- `experiment_data/`
- `resource/video/`
- Project media and demo exports already hosted on the project website.
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
        ...
```

If the artifacts need to be shared through GitHub, use a separate artifacts
repository and clone it only when needed. A private or public artifact repo can
also be added as an optional submodule later, but normal users should not need
it for installation or simulation.

## Adding New Media

Do not add videos or generated media to the main repository. Put qualitative
demos on the project website and keep raw/source exports in a local or separate
artifact checkout.
