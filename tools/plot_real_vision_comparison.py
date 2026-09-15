#!/usr/bin/env python3
"""Render the recorded real-vision comparison; never connects to hardware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = ("sugar_box", "cube_40mm", "charger", "apple", "can")
LABELS = ("Sugar box", "Cube", "Charger", "Apple", "Can")
REFLEX = (
    "artifacts/reflex_ten_runs/20260913T230536_911538Z/summary.csv",
    "artifacts/reflex_ten_runs/20260913T235549_436157Z/corrected_summary.csv",
    "artifacts/reflex_ten_runs/20260914T003241_264161Z/summary.csv",
    "artifacts/reflex_ten_runs/20260914T021908_736921Z/summary.csv",
    "artifacts/reflex_can_20260914/manual_results/summary.csv",
)
GGX = "artifacts/ggx_paper_results/matched_five_user_table_01/results.csv"
RASPBERRY_REFLEX = "artifacts/reflex_ten_runs/20260914T214642_549291Z/summary.json"
RASPBERRY_FILTER = "artifacts/ggx_raspberry_20260914/official_filter/summary.json"
NUT_DIAGNOSTIC = "artifacts/reflex_hex_nut_20260914/minimum_width_height5_02/summary.json"
NUT_NOTE = """Hex nut (12 mm across flats, 11 mm high) is exploratory and excluded
from success-rate denominators: the current Reflex minimum-width / 5 mm-center
configuration was rejected offline for desk penetration at a representative
location, not evaluated across ten localized starts. GGX was not tested.
Neither method has a physical nut attempt recorded here.
"""
DISPLAY_ORDER = ("sugar_box", "can", "apple", "cube_40mm", "charger")
SIX_OBJECT_NOTE = """Raspberry uses the final post-tuning Reflex run. Its GGX
top-100 candidates were all rejected by the official 20 mm scene-distance
filter: the ten planned starts therefore have no success, with zero physical
attempts. This is one shared pool-selection event, not ten independent
generation or physical failures. Reflex uses a calibrated desk plane with
zero extra clearance for Raspberry; collision/contact policies are not identical.
"""
SINGLE_CAPTION = """Real-vision grasp comparison across five objects. Bars show
end-to-end success over ten planned starts per object (50 per method overall),
including pre-execution rejection as an unsuccessful outcome. Reflex and GGX
share reference perception and object-relative starts, while retaining
method-specific execution and contact policies. Outcomes are operator-reported;
Cube P4 uses the corrected original failure, and Can Reflex includes retrospective
reports. Protocol details and execution-versus-rejection breakdown are reported
in the text.
"""
COMBINED_CAPTION = """Real-vision grasp comparison. (a) Shared RGB-D observations
provide the segmented target point cloud for analytical Reflex and GraspGen-X
(GGX). Reflex uses one frozen grasp template per object; GGX preserves the
original ranking of a frozen top-100 candidate pool. Selected grasps are
executed with method-specific control and contact policies, followed by an
80 mm lift and 3 s hold. (b) End-to-end success across five objects, with ten
planned starts per object and 50 per method overall. Pre-execution rejection
counts as no success; not all planned combinations were physically executed.
The source records include the corrected original Cube P4 failure and
retrospective operator reports for Can Reflex. See the text for protocol
differences and failure-stage breakdown.
"""
CAPTION = """Shared-perception real-robot grasp evaluation. (a) The analytical
thumb-reflex strategy and GraspGen-X (GGX) use the same saved RGB-D observations
and segmented target point clouds. Reflex freezes one grasp template per object;
GGX freezes the original top-100 ranked candidates and selects an eligible
candidate for each start. Object-relative Cartesian starts P1–P10 and the
80 mm lift / 3 s hold criterion are shared. (b) End-to-end success across ten
planned starts per object. (c) Paired per-start outcomes (R: Reflex; G: GGX).
Filled cells denote operator-reported success; blank cells denote no success,
including pre-execution rejection, operator veto, and unsuccessful execution.
Thus, the 100 planned method–object–start combinations are not 100 physically
executed grasps. Method-specific contact and execution policies are retained;
the historical sugar-box Reflex run uses a different wrist branch from GGX.
Hand commands are open-loop because reliable hand feedback is unavailable.
Cube P4 uses the corrected original failure; the separate repeat is excluded.
Can Reflex outcomes are retrospectively operator-reported, with incomplete
execution logs. These descriptive results do not establish simulation prediction
accuracy or attribute all deployment failures to the grasp generator.
"""


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_results(root: Path, include_raspberry: bool = False):
    points = {f"P{i}" for i in range(1, 11)}
    data = {"Reflex": [], "GGX": []}
    ggx_rows = read_rows(root / GGX)
    if len(ggx_rows) != 10 or {r["point_id"] for r in ggx_rows} != points:
        raise ValueError("GGX table must contain exactly P1–P10")
    ggx_by_point = {r["point_id"]: r for r in ggx_rows}
    for obj, source in zip(OBJECTS, REFLEX):
        rows = read_rows(root / source)
        if len(rows) != 10 or {r["point_id"] for r in rows} != points:
            raise ValueError(f"Expected exactly P1–P10: {source}")
        indexed = {r["point_id"]: r for r in rows}
        reflex, ggx = [], []
        for i in range(1, 11):
            row = indexed[f"P{i}"]
            outcome = row.get("operator_outcome", row.get("reported_outcome"))
            if outcome not in {"success", "fail", "not_executed"}:
                raise ValueError(f"Unresolved outcome in {source}: {outcome}")
            reflex.append(int(outcome == "success"))
            value = ggx_by_point[f"P{i}"][obj]
            if value not in {"0", "1"}:
                raise ValueError(f"Nonbinary GGX outcome: {value}")
            ggx.append(int(value))
        data["Reflex"].append(reflex)
        data["GGX"].append(ggx)
    if include_raspberry:
        report = json.loads((root / RASPBERRY_REFLEX).read_text())
        rows = report["trials"]
        if report.get("status") != "completed" or len(rows) != 10 or {r["point_id"] for r in rows} != points:
            raise ValueError("Raspberry requires a completed ten-start Reflex record")
        indexed = {r["point_id"]: r for r in rows}
        outcomes = [indexed[f"P{i}"]["operator_outcome"] for i in range(1, 11)]
        if any(value not in {"success", "fail", "not_executed"} for value in outcomes):
            raise ValueError("Unresolved Raspberry Reflex outcome")
        filtered = json.loads((root / RASPBERRY_FILTER).read_text())
        candidates = filtered.get("candidates", [])
        if (filtered.get("status") != "completed" or len(candidates) != 100
                or {r["rank"] for r in candidates} != set(range(100))
                or any(r.get("official_scene_clear") is not False for r in candidates)):
            raise ValueError("Raspberry GGX zero-success derivation requires a fully rejected top-100 pool")
        data["Reflex"].append([int(value == "success") for value in outcomes])
        # Protocol-level no-success entries, NOT fabricated physical outcomes.
        data["GGX"].append([0] * 10)
    return data


def render(data, out: Path, dpi: int, layout: str = "single", objects=OBJECTS,
           nut_status: bool = False, nut_assigned_zero: bool = False,
           show_assigned_zero_note: bool = True):
    if nut_assigned_zero and (layout == "overview" or "hex_nut" not in objects):
        raise ValueError("Assigned-zero nut scenario requires compact layout and hex_nut column")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rh56-paper-matplotlib")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib and numpy; use .venv312/bin/python") from exc

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    blue, orange, ink = "#167D9A", "#D66A28", "#202B36"
    colors = {"Reflex": blue, "GGX": orange}
    display_order = DISPLAY_ORDER + (("raspberry",) if "raspberry" in objects else ())
    display_order += ("hex_nut",) if nut_assigned_zero else ()
    count_objects = len(objects)
    labels = dict(zip(OBJECTS, LABELS), raspberry="Raspberry")
    def rate_label(rate):
        return str(int(round(rate))) if abs(rate-round(rate)) < 1e-8 else f"{rate:.1f}"
    if layout in {"single", "combined"}:
        combined = layout == "combined"
        fig = plt.figure(figsize=(3.45, 4.45 if combined else 2.50), facecolor="white")
        if combined:
            a = fig.add_axes([.035, .565, .93, .40])
            a.set(xlim=(0, 1), ylim=(0, 1))
            a.axis("off")
            a.text(0, 1, "(a)  Shared-perception pipeline", va="top", fontsize=9,
                   weight="bold", color=ink)

            def compact_box(x, y, w, h, title, detail, color):
                a.add_patch(FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                    facecolor="#F6F8FA", edgecolor=color, linewidth=.9))
                a.text(x+w/2, y+h*.72, title, ha="center", va="center",
                       fontsize=8.5, weight="bold", color=color)
                a.text(x+w/2, y+h*.28, detail, ha="center", va="center",
                       fontsize=7.5, color=ink, linespacing=1.2)

            def compact_arrow(start, end):
                a.annotate("", xy=end, xytext=start,
                           arrowprops={"arrowstyle": "->", "lw": .85, "color": "#798590"})

            compact_box(.09, .725, .82, .165, "Shared RGB-D observation",
                        "D435 · segmented target point cloud", ink)
            compact_box(.015, .355, .44, .235, "Reflex (ours)",
                        "One frozen\ngrasp template", blue)
            compact_box(.545, .355, .44, .235, "GGX",
                        "Frozen top-100\nOriginal ranking", orange)
            compact_box(.09, .035, .82, .18, "Grasp execution & scoring",
                        "Lift 80 mm · hold 3 s", ink)
            for cx in (.235, .765):
                compact_arrow((.5, .714), (cx, .605))
                compact_arrow((cx, .342), (.5, .230))
            fig.text(.035, .537, "(b)  Score with assigned nut zeros" if nut_assigned_zero else "(b)  End-to-end success", fontsize=9,
                     weight="bold", color=ink)
        ax = fig.add_axes([.155, .10, .83, .36] if combined else [.155, .20, .83, .64])
        indices = [objects.index(obj) for obj in display_order]
        x = np.arange(count_objects + 1, dtype=float)
        if count_objects >= 6:
            x[-1] += .30  # Reserve a wider overall slot beyond the divider.
        for j, method in enumerate(("Reflex", "GGX")):
            counts = np.array([sum(data[method][i]) for i in indices])
            rates = np.append(counts * 10, counts.sum() / (10 * count_objects) * 100)
            pos = x + (j-.5)*.36
            # Decimal overall labels need more space than integer object rates:
            # keep the shorter bar's label clear of its taller neighbour.
            if count_objects >= 6:
                pos[-1] = x[-1] + (j-.5)*.70
            ax.bar(pos, rates, .33, color=colors[method], zorder=3,
                   label="Reflex (ours)" if method == "Reflex" else "GGX")
            for px, rate in zip(pos, rates):
                ax.text(px, rate+2.4, rate_label(rate), ha="center", fontsize=7 if count_objects >= 6 else 8,
                        color=colors[method], weight="bold")
        ax.set(ylim=(0, 103), yticks=[0, 20, 40, 60, 80, 100])
        ax.set_ylabel("Score (%)" if nut_assigned_zero else "End-to-end success (%)", fontsize=8.5, labelpad=3)
        tick_labels = ["Sugar\nbox", "Can", "Apple", "Cube", "Charger"]
        if "raspberry" in objects:
            tick_labels.append("Rasp-\nberry")
        if nut_assigned_zero:
            tick_labels[4] = "Char-\nger"
            tick_labels.append("Nut")
        ax.set_xticks(x, tick_labels + ["Overall"], fontsize=7 if count_objects >= 6 else 7.5)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", labelsize=8, length=2, pad=3)
        ax.spines["left"].set_color("#A9B4BF")
        ax.spines["bottom"].set_color("#A9B4BF")
        ax.grid(axis="y", color="#E7ECF0", linewidth=.6, zorder=0)
        ax.axvline(count_objects-.5, color="#B5BEC6", linewidth=.7, linestyle="--")
        ax.legend(loc="lower center", bbox_to_anchor=(.47, 1.03), frameon=False,
                  ncol=2, fontsize=8.5, handlelength=1.2, columnspacing=1.5)
        if nut_status or (nut_assigned_zero and show_assigned_zero_note):
            # Separate exploratory status from measured rates: missing != zero.
            old_height = fig.get_figheight()
            extra = .62
            fig.set_figheight(old_height + extra)
            scale = old_height / (old_height + extra)
            offset = extra / (old_height + extra)
            for axis in fig.axes:
                box = axis.get_position()
                axis.set_position([box.x0, offset + box.y0 * scale,
                                   box.width, box.height * scale])
            for label in fig.texts:
                px, py = label.get_position()
                label.set_position((px, offset + py * scale))
            fig.text(.08, offset * .76, "*Nut: assigned 0/10, not evaluated trials" if nut_assigned_zero else "Hex nut — exploratory; excluded from rates",
                     fontsize=8, weight="bold", color=ink)
            fig.text(.08, offset * .46, "Reflex: current pose rejected offline",
                     fontsize=8, color=blue)
            fig.text(.08, offset * .16, "GGX: not tested", fontsize=8, color=orange)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out / f"real_vision_comparison.{ext}", dpi=dpi, facecolor="white")
        plt.close(fig)
        return
    fig = plt.figure(figsize=(7.2, 4.7), facecolor="white")
    a = fig.add_axes([.025, .66, .95, .30])
    a.set(xlim=(0, 1), ylim=(0, 1))
    a.axis("off")
    a.text(0, .99, "(a)  Shared perception and object-relative starts", va="top",
           fontsize=10, weight="bold", color=ink)

    def box(x, y, w, h, title, detail, color):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=0.02",
                                  facecolor="#F6F8FA", edgecolor=color, linewidth=.85))
        a.text(x+w/2, y+h*.70, title, ha="center", va="center", weight="bold", color=color, fontsize=8)
        a.text(x+w/2, y+h*.29, detail, ha="center", va="center", fontsize=7.2, color=ink, linespacing=1.3)

    def arrow(start, end):
        a.annotate("", xy=end, xytext=start,
                   arrowprops={"arrowstyle": "->", "lw": .9, "color": "#798590"})

    box(.012, .24, .232, .43, "Shared RGB-D reference", "D435 · segmented point cloud\nSame input for both methods", ink)
    box(.32, .50, .302, .31, "Analytical Reflex", "One frozen grasp template", blue)
    box(.32, .085, .302, .31, "GraspGen-X", "Frozen top-100 · original ranking", orange)
    box(.70, .24, .285, .43, "P1–P10 evaluation", "XY relocalization on replacement\nLift 80 mm · hold 3 s", ink)
    for y in (.655, .24):
        arrow((.25, .455), (.308, y))
        arrow((.633, y), (.689, .455))
    a.text(.5, -.015, "Method-specific contact / execution policies; supervised execution; no reliable hand feedback.",
           ha="center", fontsize=7, color="#53616E")

    b = fig.add_axes([.080, .19, .465, .365])
    b.set_title("(b)  End-to-end success", loc="left", fontsize=10, weight="bold", pad=20)
    x = np.arange(count_objects + 1)
    for j, method in enumerate(("Reflex", "GGX")):
        successes = np.array([sum(row) for row in data[method]])
        counts = np.append(successes, successes.sum())
        denom = np.array([10]*count_objects+[10*count_objects])
        rates = counts / denom * 100
        pos = x + (j-.5)*.34
        b.bar(pos, rates, .31, color=colors[method], zorder=3)
        for px, rate, count, n in zip(pos, rates, counts, denom):
            b.text(px, rate+2.4, f"{count}/{n}", ha="center", fontsize=6.6, color=colors[method], weight="bold")
    b.set(ylim=(0, 103), yticks=[0, 20, 40, 60, 80, 100], ylabel="Success / planned starts (%)")
    b.set_xticks(x, [labels[obj].replace("Sugar box", "Sugar\nbox").replace("Raspberry", "Rasp-\nberry") for obj in objects] + ["Overall"], fontsize=7)
    b.tick_params(axis="x", length=0)
    b.tick_params(axis="y", labelsize=7)
    b.spines["left"].set_color("#BBC3CB")
    b.spines["bottom"].set_color("#BBC3CB")
    b.grid(axis="y", color="#E7ECF0", linewidth=.6, zorder=0)
    b.axvline(count_objects-.5, color="#B5BEC6", linewidth=.7, linestyle="--")
    b.legend(handles=[Patch(color=blue, label="Analytical Reflex"), Patch(color=orange, label="GGX")],
             loc="lower left", bbox_to_anchor=(-.02, 1.00), frameon=False, ncol=2, fontsize=7,
             handlelength=1.1, columnspacing=1)

    c = fig.add_axes([.655, .19, .325, .365])
    c.set_title("(c)  Outcomes by start", loc="left", fontsize=10, weight="bold", pad=20)
    c.set(xlim=(-.5, 9.5), ylim=(count_objects-.5, -.5))
    for obj in range(count_objects):
        for p in range(10):
            for j, method in enumerate(("Reflex", "GGX")):
                cy = obj + (j-.5)*.37
                value = data[method][obj][p]
                c.add_patch(Rectangle((p-.40, cy-.155), .8, .31,
                                      facecolor=colors[method] if value else "white",
                                      edgecolor=colors[method] if value else "#CFD6DC", linewidth=.6))
    c.set_xticks(range(10), [str(i) for i in range(1, 11)], fontsize=7)
    c.set_yticks(range(count_objects), [labels[obj] for obj in objects], fontsize=7)
    c.set_xlabel("Start P1–P10", fontsize=7)
    c.tick_params(length=0)
    for spine in c.spines.values():
        spine.set_visible(False)
    c.text(0, 1.04, "Upper: Reflex   /   Lower: GGX", transform=c.transAxes, fontsize=7)
    c.text(0, -.29, "Filled = success; blank = no success", transform=c.transAxes, fontsize=7, color="#53616E")

    fig.text(.025, .047, "Denominator includes pre-execution rejection; not all planned starts were physically attempted.",
             fontsize=7.1, color=ink)
    fig.text(.025, .020, "Operator-scored outcomes. Cube P4 corrected; Can Reflex includes retrospective reports. See caption for protocol differences.",
             fontsize=6.6, color="#53616E")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"real_vision_comparison.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository containing frozen source artifacts")
    parser.add_argument("--out", type=Path, help="Output directory (defaults to a layout-specific artifact directory)")
    parser.add_argument("--layout", choices=("single", "combined", "overview"), default="single")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--include-raspberry", action="store_true", help="Add final Raspberry Reflex run and GGX empty-pool outcome; preserve original five-object sources")
    parser.add_argument("--include-nut-status", action="store_true",
                        help="Append exploratory nut status, not fabricated 0/10 outcomes")
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.include_nut_status and args.layout == "overview":
        parser.error("--include-nut-status requires single or combined layout")
    if args.include_nut_status:
        nut = json.loads((args.root / NUT_DIAGNOSTIC).read_text())
        if (nut.get("status") != "requested_endpoint_desk_rejected"
                or nut.get("hardware_contacted") is not False):
            raise ValueError("Nut status requires the saved offline desk-rejected diagnostic")
    if args.out is None:
        suffix = "" if args.layout == "overview" else f"_{args.layout}"
        suffix += "_six" if args.include_raspberry else ""
        suffix += "_nut_status" if args.include_nut_status else ""
        args.out = ROOT / f"artifacts/paper_figures/real_vision_comparison{suffix}"
    objects = OBJECTS + (("raspberry",) if args.include_raspberry else ())
    data = load_results(args.root, args.include_raspberry)
    args.out.mkdir(parents=True, exist_ok=True)
    render(data, args.out, args.dpi, args.layout, objects, args.include_nut_status)
    if args.include_nut_status:
        with (args.out / "exploratory_status.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["object", "method", "status", "physical_attempts", "included_in_rates"])
            writer.writerow(["hex_nut", "Reflex", "current_configuration_offline_desk_rejected", 0, False])
            writer.writerow(["hex_nut", "GGX", "not_tested", 0, False])
    with (args.out / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["object", "method", "successes", "planned_starts", "success_rate"])
        for method, rows in data.items():
            for obj, row in zip(objects, rows):
                writer.writerow([obj, method, sum(row), 10, sum(row)/10])
            total = sum(map(sum, rows))
            denominator = 10 * len(objects)
            writer.writerow(["overall", method, total, denominator, total/denominator])
    def caption_for(text):
        text += NUT_NOTE if args.include_nut_status else ""
        if not args.include_raspberry:
            return text
        return (text.replace("five objects", "six objects").replace("50 per method", "60 per method")
                .replace("100 planned", "120 planned").replace("100 physically", "120 physically")
                + SIX_OBJECT_NOTE)
    sources = (*REFLEX, GGX) + ((RASPBERRY_REFLEX, RASPBERRY_FILTER) if args.include_raspberry else ())
    sources += (NUT_DIAGNOSTIC,) if args.include_nut_status else ()
    metadata = {"schema": "rh56_real_vision_figure/v1", "layout": args.layout,
                "display_order": list(objects if args.layout == "overview" else DISPLAY_ORDER + (("raspberry",) if args.include_raspberry else ())),
                "outcome_object_order": list(objects), "sources_sha256": {
        path: hashlib.sha256((args.root / path).read_bytes()).hexdigest()
        for path in sources}, "outcomes": data, "assumptions": caption_for(CAPTION).splitlines()}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    captions = {"single": SINGLE_CAPTION, "combined": COMBINED_CAPTION, "overview": CAPTION}
    (args.out / "caption.txt").write_text(caption_for(captions[args.layout]))
    print("Read-only plotting: operator-reported outcomes; denominator includes unexecuted starts.")
    print(f"Saved PDF, editable SVG, PNG, source hashes, summary.csv and caption: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
