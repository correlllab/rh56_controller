# RH56 15-object grasp-success reproduction

This experiment is intended to reproduce the 15-object, 10-start-pose grasp
benchmark in Fig. 5 and Fig. 9 of the RH56DFX paper, and to compare the paper's
analytical iterative controller with a pretrained GraspGen-X baseline.

## Primary comparison

The paper-facing matrix is:

- 15 objects: 10 rigid YCB/YCB-like objects and 5 delicate objects;
- 10 approach poses per object;
- 150 trials per method and 300 trials for Iterative versus GraspGen-X;
- the same object pose, nominal approach pose, lift trajectory, hold interval,
  friction assumptions, and success rule for both methods.

The trial succeeds only if the hand commands a 200 mm vertical lift at
0.1 m/s, retains opposing contact, clears the table, and still holds the object
at the end. The simulation uses a 180 mm final-lift threshold as a documented
contact-dynamics tolerance. Damage is a separate required failure condition for
the five delicate objects, but it cannot be inferred from a rigid MuJoCo proxy.

## Ten nominal approach poses

The existing collision-test grid matches the nominal structure shown in Fig. 5.
With the default `y-` approach axis, offsets are relative to the final local
pre-grasp pose:

| Point | Approach offset (mm) | Lateral offset (mm) | Height offset (mm) |
|---|---:|---:|---:|
| P1 | 250 | 0 | 0 |
| P2 | 0 | 0 | 250 |
| L1_d-150 | 250 | -150 | 100 |
| L1_d-50 | 250 | -50 | 100 |
| L1_d+50 | 250 | +50 | 100 |
| L1_d+150 | 250 | +150 | 100 |
| L2_d-150 | 250 | -150 | 250 |
| L2_d-50 | 250 | -50 | 250 |
| L2_d+50 | 250 | +50 | 250 |
| L2_d+150 | 250 | +150 | 250 |

The paper also says that lateral offsets within +/-30 mm and orientation around
the approach axis were randomized. It does not publish the realized poses,
random seed, or orientation distribution. The current runner therefore uses
the nominal centers with a fixed orientation and records this difference in
`assumptions.json`; it does not invent unreported random samples.

## What is implemented

[`run_paper_15_object_grasp_success.py`](../tools/run_paper_15_object_grasp_success.py)
implements:

- the exact nominal P1-P10 grid from the earlier collision experiment;
- Iterative width-space closure about a fixed grasp point, including the RH56
  wrist compensation required by the coupled joints;
- method-independent approach, approximate 6 N MuJoCo contact-force control,
  200 mm lift, hold, and success/failure classification;
- pretrained GraspGen-X candidate loading and selection without using trial
  outcomes to choose a grasp;
- process-parallel trials, per-trial CSV, object/group/point summaries, plots,
  candidate hashes, and an assumptions file.

The important distinction is that collision feasibility is only an approach
diagnostic. It is not counted as grasp success unless the dynamic lift-and-hold
sequence also succeeds.

## Current validation status

The complete dynamics pipeline has been smoke-tested at P2. With the shared
contact controller, GraspGen-X retained the bottle and can proxies through the
lift, while the Iterative controller retained the egg proxy. This is evidence
that approach, closure, contact sensing, force control, lift, hold, and scoring
are connected correctly.

It is not yet evidence for a publishable 15-object rate. The current objects are
estimated boxes, cylinders, and spheres. A solid cylinder cannot reproduce a
bottle neck or cup rim, and the delicate proxies cannot represent crushing or
puncture. A plane3/plane4/plane5 sweep on the bottle, can, and sugar-box proxies
also showed that changing finger count alone does not remove this geometry
confound.

The pretrained candidate cache currently contains an executable open-hand
candidate for 7 of the 15 primitive proxies. The other eight need a larger
inference sample, a better mesh, or both. `no_executable_candidate` is reported
as a baseline failure and is never silently dropped.

## Commands

Export the explicitly non-measured primitive meshes used only for pipeline
development:

```bash
.venv312/bin/python tools/export_paper_object_proxy_meshes.py
```

Generate GraspGen-X candidates in the GraspGen-X environment:

```bash
/home/tanxuan/workspace/GraspGenX/.venv/bin/python \
  tools/run_graspgenx_paper_object_inference.py \
  --topk 1000
```

Run a short, clearly exploratory P2 pilot:

```bash
.venv312/bin/python tools/run_paper_15_object_grasp_success.py \
  --objects paper_bottle paper_can paper_sugar_box paper_egg \
  --points P2 \
  --workers 8 \
  --out artifacts/paper_15_object_grasp_success/pilot_p2
```

Render the learned target poses separately from the shared RH56 motion and
force-control executor. The default output contains individual videos for the
bottle, can, sugar-box, and egg proxies plus a synchronized four-panel montage:

```bash
.venv312/bin/python tools/render_graspgenx_paper_object_video.py
```

The overlay reports the selected confidence rank, model score, execution phase,
object lift, lateral displacement, and opposing normal forces. This makes a
high-confidence but dynamically poor target pose visible instead of treating
candidate confidence as grasp success.

For small-object diagnosis, audit all cached candidates before dynamics:

```bash
.venv312/bin/python tools/analyze_graspgenx_table_clearance.py
```

The current 20 mm nut proxy has zero collision-free open-hand targets among its
100 cached candidates. To visualize the top-ranked rejected target without
mistaking it for an accepted grasp, run:

```bash
.venv312/bin/python tools/render_graspgenx_paper_object_video.py \
  --objects paper_nut \
  --point P2 \
  --fallback-rank 0 \
  --out artifacts/paper_15_object_grasp_success/graspgenx_small_object_diagnostic
```

Fallback targets are always scored as failures and labeled as rejected in the
video. This test diagnoses support-surface clearance; it does not establish a
physical M6-nut success rate because the present nut is still a 20 mm box proxy.

After replacing the proxies with measured assets and defining delicate-object
damage limits, the intended full matrix is:

```bash
.venv312/bin/python tools/run_paper_15_object_grasp_success.py \
  --methods iterative graspgenx \
  --workers 8 \
  --out artifacts/paper_15_object_grasp_success/results
```

Do not present the full command's default primitive-proxy rates as a reproduction
of the physical experiment.

## Inputs still required for the formal run

For every object, collect the fields in
[`paper_15_object_asset_requirements.csv`](paper_15_object_asset_requirements.csv):

- watertight collision mesh or a validated multi-primitive model;
- measured mass, center of mass, and inertia;
- table pose and mesh scale;
- major axis, grasp width, grasp point/plane, and active-finger mode;
- rigid-object force target or delicate-object force and damage limits;
- the original random pose samples/seed if an exact rather than structural
  reproduction is required.

The runner must be connected to those assets before collecting the final 300
trials. Raw trial rows and failures should be retained even when a method has no
executable grasp candidate.
