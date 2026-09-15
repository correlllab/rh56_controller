# Calibrating the geometric model against the hardware rates — 2026-09-10

Four inputs were corrected or measured. Three are model inputs and were wrong;
the fourth is the support-plane tolerance, which was previously chosen to
reproduce the hardware rates and has now been measured against the hand's own
collision geometry.

| | Naive | Iterative | Reflex |
|---|---|---|---|
| **Model, measured support-plane geometry, no free parameter** | **48.0%** | 71.3% | **90.7%** |
| **Hardware, 150 trials per method** | **48.0%** | 82.0% | **86.7%** |
| Hardware 95% CI | 39.8–56.3% | 74.9–87.8% | 80.2–91.7% |

Naive closure lands on the hardware rate exactly and reflex closure is 4.0 points
high; both fall inside the hardware confidence interval. Iterative closure is
10.7 points low, and the residual is 41 support-plane collisions whose depth
distribution is given below.

Two earlier versions of this file reported different numbers. The first was
48.0 / 82.0 / 90.7 and called the support-plane tolerance "the one fitted
parameter"; that tolerance was 10 mm and the measured value is 3.4 mm. The
second was 48.0 / 68.0 / 84.7, computed with the small screwdriver's handle
height set equal to the width it is pinched across.

## The two input corrections

**Object cross-sections.** The proxies were first-pass estimates. Measured
values for six objects: raspberry 20 mm (unchanged), small screwdriver 22→15,
pen 12→15, charger 60→35, big screwdriver 35→25, strawberry 30→25. Five
of the six shrink, which lowers their grasp points and moves two more objects
below the height at which the naive terminal pose misses the object entirely.

**Small screwdriver handle height.** Taking the measured 15 mm as the whole
proxy put the screwdriver's grasp point at 7.5 mm and took it to 0/10 on every
strategy, against 0/5/9 for the raspberry in the same size class. The 15 mm is
the width the handle is pinched across, not its height off the table. Sweeping
the height gives a cliff between 10.0 and 11.0 mm of grasp height and a plateau
from 11 to 15 mm returning exactly the raspberry's 0/5/9; above 17.5 mm it
reaches 0/9/9. The proxy is now 145×15×24 mm, grasp point 12 mm, mid-plateau.
Worth 5 iterative and 9 reflex trials of the 150.

| handle height | grasp point | naive | iterative | reflex |
|---|---|---|---|---|
| 15 mm | 7.5 mm | 0/10 | 0/10 | 0/10 |
| 20 mm | 10.0 mm | 0/10 | 0/10 | 0/10 |
| **24 mm (chosen)** | **12.0 mm** | **0/10** | **5/10** | **9/10** |
| 30 mm | 15.0 mm | 0/10 | 5/10 | 9/10 |
| 35 mm | 17.5 mm | 0/10 | 9/10 | 9/10 |

**Approach conditions.** The hardware protocol samples ten approach poses per
object with lateral jitter of ±3 cm and exactly one adversarial pose level with
the grasp point. The reconstruction used ±150 mm of lateral offset and put the
eight elevated conditions at 100 and 250 mm. Lateral jitter is now ±30 mm as
stated, and the elevated conditions sit at 150 and 250 mm.

Same model, same tolerances, all four combinations, at the 10 mm tolerance the
table was originally built with:

| Object sizes | Approach conditions | Naive | Iterative | Reflex |
|---|---|---|---|---|
| estimated | ±150 mm, 100/250 mm | 48.7% | 65.3% | 77.3% |
| measured | ±150 mm, 100/250 mm | 42.0% | 65.3% | 82.7% |
| estimated | ±30 mm, 150/250 mm | 60.0% | 84.7% | 90.7% |
| measured | ±30 mm, 150/250 mm | 48.0% | 84.7% | 90.7% |

The approach conditions carry iterative closure: 65.3% to 84.7% either way the
object sizes are set, because the ±150 mm starts required a diagonal transit that
drove the fingers through the support plane on the way in. The object sizes
carry naive closure: they pull it back from 60.0% to 48.0% by moving the
strawberry and the charger below the height at which a wrist held at the
open-hand pose no longer lands its contacts on the object.

## Measuring the support-plane tolerance

`code/calibrate_capsule_envelope.py` compares two lowest points of the hand in
the same world frame, over all 41,250 hand states the three strategies command
across the 15 objects and 10 approach conditions:

    z_capsule   lowest point of the 13-capsule proxy
    z_mesh      lowest point of the model's own collision geometry
                (palm boxes and cylinder, phalanx STL meshes)

A capsule test with tolerance τ and a mesh test with no tolerance agree exactly
when τ equals `z_mesh − z_capsule`, so that difference is the tolerance.

| | Median | p05 | p95 | Range |
|---|---|---|---|---|
| All 41,250 states | 4.20 mm | | | 0.32–6.89 mm |
| Within 25 mm of the table (22,833 states) | **3.43 mm** | 1.78 mm | 6.79 mm | |
| — naive closure | 5.93 mm | 4.20 mm | 6.79 mm | |
| — iterative closure | 3.40 mm | 3.00 mm | 6.81 mm | |
| — thumb reflex | 3.03 mm | 0.32 mm | 3.11 mm | |

The capsule proxy encloses the collision geometry at every one of the 41,250
states, so the tolerance is one-sided and the capsule test is never optimistic.
The binding capsule is a fingertip in almost every case — `thumb_3`, `middle_1`,
`index_1` — and the gap is close to the difference between a 8–10 mm capsule
radius and the actual distal phalanx.

Two checks on the measurement. The gap depends on the hand's configuration and
its wrist pitch but not on its yaw, since the wrist rotation is `Rz·Ry` and `Rz`
leaves the world vertical fixed; sampling six yaws gives a spread below 1 nm.
And MuJoCo collides a mesh as its convex hull, which could make the collision
geometry a loose stand-in for the part, so the same measurement was repeated
against the undecimated visual meshes: the two agree to a median of −0.11 mm
with a p05–p95 range of −0.31 to 0.20 mm. The collision geometry is the part.

## What the tolerance is worth

| Support-plane geometry | Naive | Iterative | Reflex |
|---|---|---|---|
| capsule, τ = 1.8 mm (measured p05) | 48.0% | 68.0% | 90.7% |
| capsule, τ = 3.4 mm (measured median) | 48.0% | 68.0% | 90.7% |
| **collision mesh, no tolerance** | **48.0%** | **71.3%** | **90.7%** |
| capsule, τ = 6.8 mm (measured p95) | 48.0% | 79.3% | 90.7% |
| capsule, τ = 10 mm (previously fitted) | 48.0% | 84.7% | 90.7% |
| Hardware | 48.0% | 82.0% | 86.7% |

Naive closure sits at 48.0% in every row, and so does reflex closure at 90.7%.
Of naive's 78 failed trials, 60 are terminal misalignments, which no clearance
tolerance can remove, and the 76 support-plane collisions that overlap them have
a median depth of 7.9 mm, well beyond the measured envelope. Its rate is set by the grasp-point heights of the
object set and not by the collision proxy, so the model reproducing 48.0%
exactly is not a consequence of any tuned quantity.

The 10 mm tolerance is 1.5 times the p95 of the measured envelope. It is what
produced the earlier exact agreement on iterative closure, and it moves nothing
else: naive and reflex closure hold 48.0% and 90.7% across the whole range.

## Where the remaining 10.7 points on iterative closure are

Under the mesh test iterative closure fails 43 of 150 trials: 41 support-plane
collisions and 10 premature object contacts, with overlap. The depth of those
support-plane collisions is bimodal:

| Strategy | Table collisions | p25 | Median | p75 | ≤3 mm |
|---|---|---|---|---|---|
| Naive | 76 | 5.63 mm | 7.87 mm | 8.14 mm | 1 |
| Iterative | 41 | 1.65 mm | 4.05 mm | 71.94 mm | 13 |
| Reflex | 12 | 86.04 mm | 100.90 mm | 114.00 mm | 0 |

Thirteen of iterative closure's 41 support-plane collisions are grazes no deeper
than 1.7 mm, at which depth a silicone fingertip deflects rather than the grasp
failing, while its upper quartile is past 70 mm. Reflex closure has no shallow
population at all: its twelve collisions run from 51.5 mm to 118.3 mm, a hand
that cannot reach the pose rather than one that brushes the table.

Forgiving support-plane contact below a depth threshold, without changing the
geometry:

| Grazes forgiven below | Naive | Iterative | Reflex |
|---|---|---|---|
| 0 mm | 48.0% | 71.3% | 90.7% |
| 2 mm | 48.0% | 79.3% | 90.7% |
| 5 mm | 48.0% | 84.7% | 90.7% |
| 10 mm | 54.0% | 90.0% | 90.7% |
| Hardware | 48.0% | 82.0% | 86.7% |

The threshold moves iterative closure alone, because naive and reflex closure
have no shallow collisions to forgive. Between 2 and 5 mm iterative closure
covers 79.3% to 84.7%, bracketing the measured 82.0%. The primary configuration
still forgives nothing and reports iterative closure 10.7 points low, because
the hypothesis behind the threshold is untested.

## What is still open

**The graze hypothesis is untested.** The claim above is that hardware tolerates
sub-2 mm fingertip contact with the table during closure and the model does not.
Testing it needs the per-trial failure cause from the hardware benchmark: if
iterative closure's 27 hardware failures are dominated by object displacement
rather than by aborted approaches, the hypothesis is wrong. Nothing in the
current archive records hardware failure causes.

**Fingertip compliance is not modelled.** The collision geometry is the rigid
part. Adding the silicone pad would move the fingertip surface outward by 1–2 mm
and make the model *more* pessimistic, not less, so it cannot close the gap; it
would however change which capsule binds. `--radius-scale` sweeps the capsule
radii but not the mesh.

**The approach path is not the cause.** Rerunning the mesh test with
`--approach-path via-overhead`, which travels at the start height until above
the pre-grasp and then descends instead of moving in a straight line, gives
48.0 / 71.3 / 90.7 — identical to the direct path in all 450 trials. Every
support-plane collision happens during closure, not during transit.

## Reproducing

    uv run python code/calibrate_capsule_envelope.py --out results/calibration
    uv run python code/run_reachability_sweep.py --sampler paper-conditions \
        --out results --tag paper_conditions_meshfloor \
        --contact-policy contact-fingers --approach-samples 24 --closure-samples 24 \
        --floor-model mesh --floor-tolerance-mm 0
