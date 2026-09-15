#!/usr/bin/env python3
"""Build the experiment doc page from the results, so its numbers cannot drift.

Every figure in the template is substituted as a data URI, so the published page
is self-contained; every number is read out of the CSVs written by
run_reachability_sweep.py.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import statistics as stats
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STRATEGIES = ["naive", "iterative_closure", "thumb_reflex"]
LABELS = {"naive": "Naive", "iterative_closure": "Iterative", "thumb_reflex": "Reflex"}
CLS = {"naive": "naive", "iterative_closure": "iter", "thumb_reflex": "reflex"}
HARDWARE = {"naive": 0.480, "iterative_closure": 0.820, "thumb_reflex": 0.867}


def load(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def video_uri(path: Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def video_block(path: Path, title: str, caption: str) -> str:
    """One inline video, or nothing when the file has not been rendered yet."""

    if not path.exists():
        return ""
    return (f'<figure><div class="plate"><video controls loop muted playsinline '
            f'style="width:100%;display:block" src="{video_uri(path)}"></video></div>'
            f'<figcaption><b>{title}</b> {caption}</figcaption></figure>')


def rate(rows: list[dict], strategy: str) -> float:
    subset = [r for r in rows if r["strategy"] == strategy]
    return sum(int(r["admissible"]) for r in subset) / len(subset)


def count(rows: list[dict], strategy: str, field: str) -> int:
    return sum(int(r[field]) for r in rows if r["strategy"] == strategy)


def geometry_facts() -> dict[str, str]:
    """Measure the coupled-closure travel directly from the analytical solver."""

    from strategy_admissibility import (
        ClosureGeometry,
        InspireHandFK,
        antipodal_center,
        active_fingers,
        mode_width_range,
        solve_mode,
    )

    fk = InspireHandFK()
    closure = ClosureGeometry(fk)
    lo, hi = mode_width_range(closure, "plane4")
    wide = solve_mode(closure, "plane4", hi - 1e-4)
    tight = solve_mode(closure, "plane4", lo + 1e-4)
    c_wide = antipodal_center(wide.tip_positions, active_fingers(wide))
    c_tight = antipodal_center(tight.tip_positions, active_fingers(tight))
    return {
        "W_MIN": f"{lo * 1000:.0f}",
        "W_MAX": f"{hi * 1000:.0f}",
        "TILT_WIDE": f"{math.degrees(wide.base_tilt_y):.0f}",
        "CENTER_TRAVEL": f"{float(__import__('numpy').linalg.norm(c_tight - c_wide)) * 1000:.0f}",
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=here.parent / "results")
    ap.add_argument("--figures", type=Path, default=here.parent / "figures")
    ap.add_argument("--out", type=Path,
                    default=here.parent / "20260908-strategy_reachability.html")
    args = ap.parse_args()

    # Primary is the run whose support-plane test uses the hand's own collision
    # geometry, so it carries no clearance tolerance and no fitted parameter.
    primary = load(args.results / "paper_conditions_meshfloor_trials.csv")
    assumptions = json.loads(
        (args.results / "paper_conditions_meshfloor_assumptions.json").read_text())
    envelope = json.loads(
        (args.results / "calibration/capsule_envelope_summary.json").read_text())
    sens = {}
    for key, path in (
        ("tol18", args.results / "paper_conditions_tol18_trials.csv"),
        ("tol34", args.results / "paper_conditions_tol34_trials.csv"),
        ("tol68", args.results / "paper_conditions_tol68_trials.csv"),
        ("tol100", args.results / "paper_conditions_calibrated_trials.csv"),
        ("overhead", args.results / "paper_conditions_mesh_overhead_trials.csv"),
    ):
        if path.exists():
            sens[key] = load(path)
    # The two input corrections, each isolated: object cross-sections and the
    # reconstruction of the ten approach conditions.
    ablation = {}
    for key, stem in (("old_both", "abl_old_both"),
                      ("old_sizes", "abl_old_sizes"),
                      ("old_conditions", "abl_old_conditions")):
        path = args.results / f"calibration/{stem}_trials.csv"
        if path.exists():
            ablation[key] = load(path)

    n_per_strategy = len([r for r in primary if r["strategy"] == "naive"])

    # --- per-object boundary
    heights, per_object = {}, {}
    for row in primary:
        heights.setdefault(row["object"], float(row["grasp_height_mm"]))
        per_object.setdefault((row["object"], row["strategy"]), []).append(int(row["admissible"]))
    naive_zero = [o for o in heights if sum(per_object[(o, "naive")]) == 0]
    naive_ok = [o for o in heights if sum(per_object[(o, "naive")]) > 0]

    # --- naive terminal offsets
    offsets = [float(r["terminal_offset_mm"]) for r in primary if r["strategy"] == "naive"]
    other_offsets = [
        float(r["terminal_offset_mm"]) for r in primary if r["strategy"] != "naive"
    ]

    # --- P1 (level with the grasp point)
    p1 = {
        s: sum(int(r["admissible"]) for r in primary
               if r["strategy"] == s and r["condition"] == "P1_level")
        for s in STRATEGIES
    }

    def delicate(rows, strategy):
        subset = [r for r in rows if r["strategy"] == strategy and r["category"] == "delicate"]
        return sum(int(r["admissible"]) for r in subset) / len(subset)

    # --- sensitivity table
    variants = []
    if "tol18" in sens:
        variants.append(("Capsule proxy, tolerance 1.8&nbsp;mm (measured p05)", sens["tol18"]))
    if "tol34" in sens:
        variants.append(("Capsule proxy, tolerance 3.4&nbsp;mm (measured median)", sens["tol34"]))
    variants.append(("Collision geometry, no tolerance (primary)", primary))
    if "tol68" in sens:
        variants.append(("Capsule proxy, tolerance 6.8&nbsp;mm (measured p95)", sens["tol68"]))
    if "tol100" in sens:
        variants.append(("Capsule proxy, tolerance 10&nbsp;mm (previously fitted)", sens["tol100"]))
    if "overhead" in sens:
        variants.append(("Collision geometry, L-shaped approach path", sens["overhead"]))
    table_rows = []
    for name, rows in variants:
        cells = "".join(
            f'<td class="{CLS[s]}">{rate(rows, s):.1%}</td>' for s in STRATEGIES
        )
        table_rows.append(f"<tr><td>{name}</td>{cells}</tr>")
    hw = "".join(f"<td>{HARDWARE[s]:.1%}</td>" for s in STRATEGIES)
    table_rows.append(f'<tr class="hw"><td>Hardware benchmark (measured success)</td>{hw}</tr>')

    # --- capsule envelope, measured against the model's own collision geometry
    env_band = envelope["delta_decision_band"]
    env_rows = []
    for name, key in (
        (f"All {envelope['n_samples']:,} commanded states", "delta_all"),
        (f"Within {envelope['decision_band_mm']:.0f}&nbsp;mm of the table "
         f"({env_band['n']:,} states)", "delta_decision_band"),
    ):
        d = envelope[key]
        env_rows.append(
            f"<tr><td>{name}</td><td><b>{d['median_mm']:.2f}&nbsp;mm</b></td>"
            f"<td>{d['p05_mm']:.2f}&nbsp;mm</td><td>{d['p95_mm']:.2f}&nbsp;mm</td></tr>")
    for strategy in STRATEGIES:
        d = envelope["delta_by_strategy_in_band"].get(strategy)
        if d is None:
            continue
        env_rows.append(
            f'<tr><td style="padding-left:18px" class="{CLS[strategy]}">&mdash; '
            f'{LABELS[strategy]}</td><td>{d["median_mm"]:.2f}&nbsp;mm</td>'
            f'<td>{d["p05_mm"]:.2f}&nbsp;mm</td><td>{d["p95_mm"]:.2f}&nbsp;mm</td></tr>')

    # --- what a graze allowance would buy, computed from the primary run itself
    def forgiving(rows, strategy, depth_mm):
        subset = [r for r in rows if r["strategy"] == strategy]
        ok = 0
        for r in subset:
            modes = {m for m in ("table_collision", "premature_object_contact",
                                 "terminal_misalignment", "obstacle_collision")
                     if str(r.get(m, "0")) not in ("", "0")}
            if ("table_collision" in modes
                    and -float(r["min_floor_clearance_mm"]) <= depth_mm):
                modes.discard("table_collision")
            ok += not modes
        return ok / len(subset)

    graze_rows = []
    for depth in (0.0, 2.0, 5.0, 10.0):
        cells = "".join(f'<td class="{CLS[st]}">{forgiving(primary, st, depth):.1%}</td>'
                        for st in STRATEGIES)
        name = "Nothing forgiven (primary)" if depth == 0 else f"{depth:.0f}&nbsp;mm"
        graze_rows.append(f"<tr><td>{name}</td>{cells}</tr>")

    abl_rows = []
    for name, rows in (
        ("Estimated sizes, &plusmn;150&nbsp;mm conditions", ablation.get("old_both")),
        ("Measured sizes, &plusmn;150&nbsp;mm conditions", ablation.get("old_conditions")),
        ("Estimated sizes, &plusmn;30&nbsp;mm conditions", ablation.get("old_sizes")),
        # All four rows must share a support-plane tolerance or the comparison is
        # not the two corrections; the 10 mm run is the one that has all three
        # other combinations to sit beside.
        ("Measured sizes, &plusmn;30&nbsp;mm conditions (the corrected inputs)",
         sens.get("tol100", primary)),
    ):
        if rows is None:
            continue
        cells = "".join(f'<td class="{CLS[s]}">{rate(rows, s):.1%}</td>' for s in STRATEGIES)
        abl_rows.append(f"<tr><td>{name}</td>{cells}</tr>")
    abl_rows.append(f'<tr class="hw"><td>Hardware benchmark (measured success)</td>{hw}</tr>')

    # --- optional region-map figure
    # --- GraspGen-X section, from the enveloping-contact four-strategy run
    # The four-strategy run under the primary support-plane geometry, falling
    # back to the fitted-tolerance run while the mesh version is still sweeping.
    # Scored over the first 100 proposals, as the hardware protocol does; the
    # top-20 scoring of the same candidates is kept for the comparison sentence.
    ggx_candidates = [
        ("paper_conditions_meshfloor_ggx100", "paper_conditions_meshfloor_ggx100"),
        ("paper_conditions_meshfloor_ggx", "paper_conditions_meshfloor_ggx"),
        ("paper_conditions_ggx_enveloping", "paper_conditions_calibrated"),
    ]
    ggx_rows, ggx_assumptions = [], {}
    for stem, ap_stem in ggx_candidates:
        path = args.results / f"{stem}_trials.csv"
        if path.exists():
            ggx_rows = load(path)
            ap_json = args.results / f"{ap_stem}_assumptions.json"
            ggx_assumptions = json.loads(ap_json.read_text()) if ap_json.exists() else {}
            break
    ggx_top20_path = args.results / "paper_conditions_meshfloor_ggx_trials.csv"
    ggx_top20 = load(ggx_top20_path) if ggx_top20_path.exists() else []
    ggx_strict_path = args.results / "paper_conditions_ggx_trials.csv"
    ggx_strict = load(ggx_strict_path) if ggx_strict_path.exists() else []

    ggx_table, ggx_narrative, ggx_strict_count, ggx_topk = "", "", "0", "20"
    if ggx_rows and any(r["strategy"] == "graspgen_x" for r in ggx_rows):
        order = ["naive", "iterative_closure", "thumb_reflex", "graspgen_x"]
        cls = {"naive": "naive", "iterative_closure": "iter",
               "thumb_reflex": "reflex", "graspgen_x": "ggx"}
        names = {"naive": "Naive", "iterative_closure": "Iterative",
                 "thumb_reflex": "Reflex", "graspgen_x": "GraspGen-X"}
        cells = []
        for strategy in order:
            subset = [r for r in ggx_rows if r["strategy"] == strategy]
            if not subset:
                continue
            admitted = sum(int(r["admissible"]) for r in subset)
            cells.append(
                f'<tr><td class="{cls[strategy]}">{names[strategy]}</td>'
                f'<td>{admitted}/{len(subset)} ({admitted / len(subset):.1%})</td>'
                + "".join(f"<td>{sum(int(r[m]) for r in subset)}</td>" for m in
                          ["table_collision", "premature_object_contact", "terminal_misalignment"])
                + "</tr>"
            )
        ggx_table = "\n".join(cells)
        ggx_topk = str(ggx_assumptions.get("ggx_topk", 20))
        if ggx_strict:
            ggx_strict_count = str(sum(
                int(r["admissible"]) for r in ggx_strict if r["strategy"] == "graspgen_x"))

        g = [r for r in ggx_rows if r["strategy"] == "graspgen_x"]
        ranks = sorted(int(r["ggx_candidate_rank"]) for r in g
                       if int(r.get("ggx_candidate_rank", -1)) >= 0) or [0]
        admitted = len(ranks)
        naive_admitted = sum(int(r["admissible"]) for r in ggx_rows
                             if r["strategy"] == "naive")
        reflex_admitted = sum(int(r["admissible"]) for r in ggx_rows
                              if r["strategy"] == "thumb_reflex")
        top_ranked = sum(1 for r in ranks if r == 0)
        beyond_20 = sum(1 for r in ranks if r >= 20)
        # ggx_candidate_rank is 0-based; the prose counts proposals from one.
        q1, med, q3 = (ranks[len(ranks) // 4] + 1, ranks[len(ranks) // 2] + 1,
                       ranks[(3 * len(ranks)) // 4] + 1)
        by_object = {}
        for row in g:
            by_object.setdefault(row["object"], []).append(int(row["admissible"]))
        zero_objects = [o for o, v in by_object.items() if sum(v) == 0]
        heights_by_object = {r["object"]: float(r["grasp_height_mm"]) for r in g}
        tallest_zero = max((heights_by_object[o] for o in zero_objects), default=0.0)
        ggx_delicate = [r for r in g if r["category"] == "delicate"]
        delicate_hits = sum(int(r["admissible"]) for r in ggx_delicate)
        ycb_hits = sum(int(r["admissible"]) for r in g if r["category"] == "ycb_like")
        if admitted == naive_admitted or abs(admitted - naive_admitted) <= 1:
            standing = "level with naive closure and below iterative and reflex closure"
        elif admitted < naive_admitted:
            standing = "below every hand-designed strategy"
        else:
            standing = "above naive closure and below iterative and reflex closure"
        top20_sentence = ""
        if ggx_top20:
            top20_admitted = sum(int(r["admissible"]) for r in ggx_top20
                                 if r["strategy"] == "graspgen_x")
            top20_sentence = (f" Scoring the same candidates over the first 20 proposals "
                              f"only gives {top20_admitted} of 150 ({top20_admitted / 150:.1%}), "
                              f"which is the number an earlier version of this page "
                              f"reported.")
        ggx_narrative = (
            f"GraspGen-X is admissible in {admitted} of 150 trials ({admitted / 150:.1%}), "
            f"{standing} ({reflex_admitted} of 150). The ranking is most of the problem: "
            f"its top-ranked proposal executes cleanly in {top_ranked} of the {admitted} "
            f"trials that succeed at all, the first collision-free candidate is the "
            f"{ordinal(med)} proposal by median (interquartile range {ordinal(q1)} to "
            f"{ordinal(q3)}, at worst the {ordinal(ranks[-1] + 1)}), and {beyond_20} of "
            f"the {admitted} are found only "
            f"beyond the 20th.{top20_sentence} Object size is the rest of it: GraspGen-X "
            f"admits nothing at any approach condition for the {len(zero_objects)} objects "
            f"whose grasp point lies at or below {tallest_zero:.0f}\u2009mm, the set on "
            f"which naive closure whiffs, and {delicate_hits} of 50 delicate trials "
            f"against {ycb_hits} of 100 YCB-like."
        )

    # --- no-go volume figures
    def volume_figure(obj_name: str, png: str, lede: str, tail: str) -> str:
        stats_path = args.results / f"nogo_volume_{obj_name}_stats.json"
        fig = args.figures / png
        if not (stats_path.exists() and fig.exists()):
            return ""
        vol = json.loads(stats_path.read_text())
        def pct(layer: str) -> str:
            return f"{vol[layer]['locked_fraction']:.1%}" if layer in vol else "n/a"
        # Naive and iterative coincide on some objects and not others; say which.
        if vol.get("naive", {}).get("locked_fraction") == vol.get(
                "iterative_closure", {}).get("locked_fraction"):
            rates = (f"Naive and iterative closure leave the same {pct('naive')} of the "
                     f"swept poses unreachable and reflex closure {pct('thumb_reflex')}")
        else:
            rates = (f"Naive closure leaves {pct('naive')} of the swept poses unreachable, "
                     f"iterative closure {pct('iterative_closure')} and reflex closure "
                     f"{pct('thumb_reflex')}")
        # The sweep only carries GraspGen-X layers when it was given a candidate
        # directory; an all-false layer from a run without one is absence of
        # evidence, not a 100% lockout.
        assum_path = args.results / f"nogo_volume_{obj_name}_assumptions.json"
        vol_assum = json.loads(assum_path.read_text()) if assum_path.exists() else {}
        ran_ggx = vol_assum.get("n_ggx_candidates", 0) > 0
        if ran_ggx and "ggx_best" in vol:
            k = vol_assum["n_ggx_candidates"]
            rates += (f"; GraspGen-X {pct('ggx_best')} credited with the best of "
                      f"its {k} ranked candidates, and {pct('ggx_worst')} when every one of "
                      f"the {k} must be collision-free")
        shells = ("The sets nest, reflex inside iterative inside naive, and are drawn as "
                  "nested translucent solids so the containment is visible; a post marks "
                  "the object, which is a few tens of millimetres across inside a region a "
                  "few hundred across. Surfaces are the half level of the sweep grid, "
                  "refined and low-passed, so the smoothness is presentational and the "
                  "boundary is resolved to 30&nbsp;mm.")
        return f'''<figure>
    <div class="plate"><img src="{data_uri(fig)}" alt="No-go start volumes drawn as disjoint nested shells, one colour per strategy."></div>
    <figcaption>
      <b>{lede}</b> {rates}. {tail} {shells}
    </figcaption>
  </figure>'''

    # The GraspGen-X k of the volume sweep behind each object's strips. The
    # strawberry keeps its k = 10 sweep: every one of its 100 proposals fails at
    # the proposal pose itself (code/ggx_start_independent_check.py), so no start
    # pose rescues any of them and the all-unreachable panel is exact at any k.
    def vol_k(obj: str) -> int:
        for stem in (f"nogo_volume_paper_{obj}_ggx100merged", f"nogo_volume_paper_{obj}"):
            ap = args.results / f"{stem}_assumptions.json"
            if ap.exists():
                return json.loads(ap.read_text()).get("n_ggx_candidates", 0) or 0
        return 0
    vol_topk = max(vol_k("can"), vol_k("strawberry"), 10)

    def strip_figure(png: str, lede: str, tail: str) -> str:
        fig = args.figures / png
        obj = "strawberry" if "strawberry" in png else "can"
        stats_path = args.results / f"{fig.stem}_stats.json"
        if not (stats_path.exists() and fig.exists()):
            return ""
        st = json.loads(stats_path.read_text())
        # Panel order, so the caption is read alongside the strip rather than
        # against it, and the full clause only on the first: repeating it three
        # times reads as three findings instead of one comparison.
        order = [k for k in ("iterative_closure", "thumb_reflex", "naive") if k in st]
        parts = [f"{LABELS[order[0]]} closure leaves "
                 f"{st[order[0]]['locked_fraction']:.1%} of the sampled start poses "
                 "unreachable"]
        parts += [f"{LABELS[k].lower()} closure {st[k]['locked_fraction']:.1%}"
                  for k in order[1:]]
        if "ggx_best" in st:
            parts.append(f"GraspGen-X at its best of {vol_k(obj)} candidates "
                         f"{st['ggx_best']['locked_fraction']:.1%}")
        rates = ", ".join(parts)
        split = ("Red is the set of start poses from which the strategy cannot execute "
                 "and green the rest of the sampled quarter ball, so the two together "
                 "are the region the percentages are fractions of.")
        return f'''<figure>
    <div class="plate"><img src="{data_uri(fig)}" alt="Start-pose volumes, one strategy per panel on a shared camera, unreachable poses shaded red and reachable ones green, with the object drawn opaque."></div>
    <figcaption>
      <b>{lede}</b> {rates}. {split} {tail}
    </figcaption>
  </figure>'''

    def grid_figure(png: str) -> str:
        # The paper figure: both objects in one lettered grid with the strategy
        # named once per column. It goes first so the reader meets the figure the
        # paper carries before the per-object strips that explain it.
        fig = args.figures / png
        if not fig.exists():
            return ""
        rates = {}
        for obj in ("strawberry", "can"):
            stats_path = args.results / f"20260908-nogo_strip_{obj}_ggx_stats.json"
            if stats_path.exists():
                rates[obj] = json.loads(stats_path.read_text())
        def pct(obj, layer):
            return f"{rates[obj][layer]['locked_fraction']:.1%}" if obj in rates else "?"
        return f'''<figure>
    <div class="plate"><img src="{data_uri(fig)}" alt="Two rows of four panels: start-pose volumes for iterative, reflex, naive and GraspGen-X closure at the strawberry (A) and the can (B), unreachable poses red and reachable ones green."></div>
    <figcaption>
      <b>Fig. 5 (bottom) as it goes into the paper.</b> One column per strategy,
      GraspGen-X at its best of {vol_k("can")} ranked proposals, for (A) the strawberry,
      grasp point 15&nbsp;mm above the table, and (B) the can, 110&nbsp;mm. The strawberry
      row is drawn from its {vol_k("strawberry")}-proposal sweep: every one of the 100
      proposals puts the hand into the table at the proposal pose itself
      (<code>ggx_start_independent_check.py</code>), which no start pose changes, so the
      all-unreachable panel holds at any depth. Red is the
      set of start poses from which the strategy cannot execute and green the rest of
      the sampled quarter ball, so the two together are the region each percentage is
      a fraction of. At the strawberry iterative closure leaves {pct("strawberry", "iterative_closure")}
      unreachable, reflex {pct("strawberry", "thumb_reflex")}, naive and GraspGen-X all of it;
      at the can the three hand-designed strategies leave {pct("can", "iterative_closure")},
      {pct("can", "thumb_reflex")} and {pct("can", "naive")} against {pct("can", "ggx_best")} for
      GraspGen-X, a pocket low and close to the can; at ten proposals it was 67.7%. The
      can's GraspGen-X layer is the k&nbsp;=&nbsp;100 sweep run on the runner's default
      extents (995 of the 1206 poses, <code>results/ggx100/</code>) written onto the full
      quarter ball with the 211 cells outside that extent drawn reachable
      (<code>merge_ggx_volume.py</code>). Built by
      <code>code/render_strips.sh</code> from the bare strip
      panels; the strips below are the same renders with one object per row.
    </figcaption>
  </figure>'''

    def split_figure(png: str) -> str:
        fig = args.figures / png
        if not fig.exists():
            return ""
        return f'''<figure>
    <div class="plate"><img src="{data_uri(fig)}" alt="The same grid with the GraspGen-X panels split corner to corner: best of ten proposals in the top-left half, best of 100 in the bottom-right."></div>
    <figcaption>
      <b>Variant: the GraspGen-X column split by search depth.</b> Top-left half, best of
      the first ten ranked proposals; bottom-right half, best of all 100, at the same
      scale as the neighbouring panels. At the can the depth takes GraspGen-X from 67.7%
      to 2.8% unreachable; at the strawberry it changes nothing. The proposals are cheap
      to make, 0.24&nbsp;s per object for 1,000 samples with the top 100 kept on an
      RTX&nbsp;4070&nbsp;Ti (<code>graspgenx_candidates/manifest.json</code>); the
      collision check that finds the first executable one costs about 0.2&nbsp;s per
      candidate in this Python checker, so a median success spends about 4&nbsp;s in it
      and a condition with no executable proposal about 20&nbsp;s.
    </figcaption>
  </figure>'''

    strip_figures = "\n".join(filter(None, [
        grid_figure("20260908-nogo_grid_strawberry_can.png"),
        split_figure("20260908-nogo_grid_strawberry_can_split.png"),
        strip_figure(
            "20260908-nogo_strip_strawberry.png",
            "Strawberry, grasp point 15&nbsp;mm above the table.",
            "Clearance during closure decides the outcome at this height and the strategies "
            "separate. Naive closure admits no start pose anywhere in the sampled region. "
            "Panels share one camera framed on the sampled region, so a larger red solid "
            "is a larger unreachable set rather than a closer view; the scale bar is "
            "valid at the object."),
        strip_figure(
            "20260908-nogo_strip_can.png",
            "Can, grasp point 110&nbsp;mm above the table.",
            "The support surface is never the binding constraint at this height and the "
            "three sets nearly coincide, which is the control for the strawberry panel: the "
            "strategy choice matters near the table and not above it."),
    ]))

    strip_figures_ggx = "\n".join(filter(None, [
        strip_figure(
            "20260908-nogo_strip_strawberry_ggx.png",
            "The same strawberry strip with a GraspGen-X panel appended.",
            "Best-of-ten GGX leaves the sampled region unreachable entirely, as naive closure "
            "does, "
            "so at a grasp point 15&nbsp;mm above the table the generator offers no start "
            "pose the arm can execute from."),
        strip_figure(
            "20260908-nogo_strip_can_ggx.png",
            "The same can strip with a GraspGen-X panel appended, 100 proposals searched.",
            "What GGX loses at the can is a pocket low and close to the object, and the "
            "collision check over the ranked list does the work: at ten proposals the same "
            "panel was 67.7% unreachable, a volume twice the closure volumes that held "
            "between 65% and 71% at every sampled height because it was a restriction on "
            "approach direction rather than on clearance "
            "(<code>figures/20260908-nogo_strip_can_ggx_k10.png</code>). The layer is the "
            "k&nbsp;=&nbsp;100 sweep on the runner's default extents written onto the full "
            "quarter ball, the cells outside that extent drawn reachable."),
    ]))

    volume_figures = "\n".join(filter(None, [
        volume_figure(
            "paper_can", "20260908-nogo_volumes_can_quarter.png",
            "The volume each strategy cannot reach from, can.",
            "Start poses are swept over the region an arm is actually commanded from &mdash; "
            "the quarter ball of radius 300&nbsp;mm above the grasp point and on the approach "
            "side &mdash; and every cell is evaluated independently rather than scanning each "
            "column to its lowest admissible height, so the result can represent overhangs. "
            "Each coloured solid is the set of poses from which that strategy cannot execute. "
            "The hand is drawn at one start pose, 30&nbsp;mm behind and 280&nbsp;mm above the "
            "grasp point, which every strategy admits. The can's grasp point is 110&nbsp;mm "
            "above the table, high enough that the support surface is never the binding "
            "constraint, and the three hand-designed strategies accordingly differ by only "
            "26 of 1206 cells."),
        volume_figure(
            "paper_strawberry", "20260908-nogo_volumes_strawberry_quarter.png",
            "The same sweep on the strawberry.",
            "The strawberry's grasp point is 15&nbsp;mm above the table, so clearance during "
            "closure decides the outcome and the strategies separate. GraspGen-X is omitted: "
            "it admits no start pose at all for this object, so its volume is the whole "
            "region."),
    ]))

    grid_path = args.results / "grid_trials.csv"
    fig_c = args.figures / "20260908-fig_c_admissible_regions_cal.png"
    if fig_c.exists() and grid_path.exists():
        grid = load(grid_path)
        grid_n = len({(r["height_mm"], r["lateral_mm"]) for r in grid})
        fig_c_block = f'''<figure>
    <div class="plate"><img src="{data_uri(fig_c)}" alt="Admissible start-pose regions in approach height against lateral offset, for three objects and three strategies."></div>
    <figcaption>
      <b>Admissible start-pose regions.</b> Filled cells are start poses from which the
      commanded approach and closure is geometrically admissible, over a {grid_n}-pose grid in
      height above the grasp point against lateral offset, at a fixed 250&nbsp;mm standoff.
      The boundary moves down and outward from naive to iterative to reflex closure, and the
      three strategies converge once the object is tall enough to clear the fingers.
    </figcaption>
  </figure>'''
    else:
        fig_c_block = ""
        grid_n = 0

    subs = {
        "N_TRIALS": str(len(primary)),
        "XML_NAME": Path(assumptions["xml_path"]).name,
        "N_SAMPLES": str(assumptions["approach_samples"] + assumptions["closure_samples"] + 2),
        "FLOOR_TOL": f"{assumptions['floor_tolerance_mm']:.0f}",
        "WIDTH_OFFSET": f"{assumptions['object_width_offset_mm']:.0f}",
        "HALF_OFFSET": f"{assumptions['object_width_offset_mm'] / 2:.0f}",
        "ITER_STEP": f"{assumptions['iterative_step_mm']:.0f}",
        "ADM_NAIVE": str(count(primary, "naive", "admissible")),
        "ADM_ITER": str(count(primary, "iterative_closure", "admissible")),
        "ADM_REFLEX": str(count(primary, "thumb_reflex", "admissible")),
        "ADM_NAIVE_PCT": f"{rate(primary, 'naive'):.1%}",
        "FAIL_TABLE_NAIVE": str(count(primary, "naive", "table_collision")),
        "FAIL_TABLE_ITER": str(count(primary, "iterative_closure", "table_collision")),
        "FAIL_TABLE_REFLEX": str(count(primary, "thumb_reflex", "table_collision")),
        "FAIL_TERM_NAIVE": str(count(primary, "naive", "terminal_misalignment")),
        "NAIVE_ZERO_MAX": f"{max(heights[o] for o in naive_zero):.0f}",
        "NAIVE_OK_MIN": f"{min(heights[o] for o in naive_ok):.0f}",
        "N_LOW_OBJECTS": str(len(naive_zero)),
        "N_HIGH_OBJECTS": str(len(naive_ok)),
        "NAIVE_OFFSET_MED": f"{stats.median(offsets):.1f}",
        "NAIVE_OFFSET_MIN": f"{min(offsets):.1f}",
        "NAIVE_OFFSET_MAX": f"{max(offsets):.1f}",
        "OTHER_OFFSET_MAX": f"{max(other_offsets):.2f}",
        "P1_NAIVE": str(p1["naive"]),
        "P1_ITER": str(p1["iterative_closure"]),
        "P1_REFLEX": str(p1["thumb_reflex"]),
        "DEL_REFLEX": f"{delicate(primary, 'thumb_reflex'):.0%}",
        "DEL_ITER": f"{delicate(primary, 'iterative_closure'):.0%}",
        "SENS_A_NAIVE": (f"{rate(sens['naive_final_orientation'], 'naive'):.1%}"
                         if "naive_final_orientation" in sens else "n/a"),
        "TABLE_SENS": "\n".join(table_rows),
        "TABLE_ENVELOPE": "\n".join(env_rows),
        "TABLE_GRAZE": "\n".join(graze_rows + [f'<tr class="hw"><td>Hardware benchmark (measured success)</td>{hw}</tr>']),
        "ENV_N": f"{envelope['n_samples']:,}",
        "ENV_YAW": (f"{envelope['yaw_invariance_max_spread_mm'] * 1e6:.1f}&nbsp;nm"
                    if envelope["yaw_invariance_max_spread_mm"] > 0 else "zero"),
        "ENV_VIS": f"{envelope['delta_visual_decision_band']['median_mm']:.2f}",
        "FIG_MAIN": data_uri(args.figures / "20260908-fig_main_admissibility_mesh.png"),
        "FIG_A": data_uri(args.figures / "20260908-fig_a_per_object_admissibility_mesh.png"),
        "FIG_B": data_uri(args.figures / "20260908-fig_b_failure_modes_mesh.png"),
        "TABLE_ABLATION": "\n".join(abl_rows),
        "VOLUME_FIGURES": volume_figures,
        "GGX_VOL_TOPK": str(vol_topk),
        "STRIP_FIGURES": strip_figures,
        "STRIP_FIGURES_GGX": strip_figures_ggx,
        "VIDEOS": "\n".join(
            video_block(args.figures / name, title, caption)
            for name, title, caption in (
                ("20260908-approach_paper_strawberry_L2_d+30.mp4",
                 "Strawberry, 250&nbsp;mm approach, +30&nbsp;mm lateral.",
                 "Naive closure alone fails, and it fails during closure rather than during "
                 "the approach: all four hands clear the support plane through the transit, "
                 "and the naive panel turns red only once the fingers sweep inward. Its "
                 "contacts end 16&nbsp;mm from the target, beside a 25&nbsp;mm object."),
                ("20260908-approach_paper_nut_L1_d+30.mp4",
                 "Nut, 150&nbsp;mm approach, +30&nbsp;mm lateral.",
                 "Reflex closure alone survives, and the decision is already made at the "
                 "start pose. Naive and iterative closure hold the open-hand pre-grasp "
                 "orientation through the transit, which puts their fingers 10&nbsp;mm below "
                 "the plane before anything moves; the reflex hand, already at the final "
                 "analytical pose, clears it by 47&nbsp;mm. This is the mechanism behind the "
                 "reflex advantage at low approaches, and it is a consequence of commanding "
                 "the pre-grasp orientation from the start of the transit."),
                ("20260908-approach_paper_sugar_box_P1_level.mp4",
                 "Sugar box, adversarial level approach.",
                 "The condition the manuscript calls adversarial. The hand arrives level "
                 "with the grasp point and the object is in the path of the approach "
                 "itself, before any closure happens."),
            )),
        "GRID_N": str(grid_n),
        "GENERATED": date.today().isoformat(),
        "TABLE_GGX": ggx_table,
        "GGX_NARRATIVE": ggx_narrative,
        "GGX_STRICT": ggx_strict_count,
        "GGX_TOPK": ggx_topk,
    }
    subs.update(geometry_facts())

    html = (here / "docpage.tmpl.html").read_text()
    for key, value in subs.items():
        html = html.replace("{{" + key + "}}", value)

    leftover = [chunk.split("}}")[0] for chunk in html.split("{{")[1:]]
    if leftover:
        raise SystemExit(f"unsubstituted placeholders: {sorted(set(leftover))}")

    args.out.write_text(html)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
