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

## Navigating Local Paper-V2 Outputs

The local `artifacts/` directory keeps a small set of canonical outputs at its
top level. Iterative debug runs, parameter sweeps, and superseded outputs live
under `artifacts/archive/`. The `artifacts/current` symlink points to the latest
paper-object pre-grasp batch, and `artifacts/latest_gifs` points directly to its
current example animations. The complete path-review video and its timestamp
index are under `artifacts/current/all_trials/`.

Useful searches from the repository root:

```bash
# List every animation, including archived iterations.
rg --files artifacts | rg '\.gif$'

# List only the current paper-object animations.
rg --files -L artifacts/current | rg '\.gif$'

# Open the complete 450-trial review video and timestamp index.
ls -lh artifacts/current/all_trials/all_trials.mp4
less artifacts/current/all_trials/trial_index.csv

# Find result tables without browsing folders manually.
rg --files artifacts | rg '(summary|assumptions)\.(csv|json)$'

# See which artifact groups use the most disk space.
du -sh artifacts/* | sort -h
```

To browse the current results over SSH, serve only the current batch:

```bash
python3 -m http.server 8765 --directory artifacts/current
```

## Adding New Media

Do not add videos or generated media to the main repository. Put qualitative
demos on the project website and keep raw/source exports in a local or separate
artifact checkout.
