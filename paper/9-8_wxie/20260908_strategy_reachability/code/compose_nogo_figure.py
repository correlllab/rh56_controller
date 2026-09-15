#!/usr/bin/env python3
"""Lay out the no-go strips of several objects as one lettered grid.

render_nogo_volume.py --strip --panels-dir writes each strip's panels bare and a
<stem>_panels.json sidecar naming them. This script stacks those strips into the
paper figure: one column per strategy with the strategy named once at the top, a
lettered row per object, the unreachable fraction over each panel, a shared
reachable/unreachable key, and a scale bar on the first panel of every row. Type
is set in the paper's serif face so the figure reads as part of the page.

A row may name a second strip; with --split LAYER that layer's panel then
holds two complete renders, the second strip's reduced and set top-left and the
row's own bottom-right, so one panel shows two search depths of the same
planner without taking more room than its neighbours.

Usage:
    uv run python compose_nogo_figure.py --panels-dir ../figures/panels \\
        --row A Strawberry 20260908-nogo_strip_strawberry_ggx \\
        --row B Can 20260908-nogo_strip_can_ggx \\
        --out ../figures/20260908-nogo_grid_strawberry_can.png
    uv run python compose_nogo_figure.py --panels-dir ../figures/panels \\
        --row A Strawberry 20260908-nogo_strip_strawberry_ggx 20260908-nogo_strip_strawberry_ggx \\
        --row B Can 20260908-nogo_strip_can_ggx 20260908-nogo_strip_can_ggx_k10 \\
        --split ggx_best --split-labels "k = 10" "k = 100" \\
        --out ../figures/20260908-nogo_grid_strawberry_can_split.png --pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Times New Roman is the IEEE body face; Liberation Serif is metric-compatible
# with it and DejaVu Serif is the last resort on a bare machine.
_SERIF = {
    False: ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    True: ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
           "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
}

INK = (20, 20, 20)


def serif(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in _SERIF[bold]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_w(font, text: str) -> float:
    return float(font.getlength(text))


def rgb255(rgb) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgb)


def load_row(panels_dir: Path, stem: str) -> tuple[dict, list[Image.Image]]:
    meta = json.loads((panels_dir / f"{stem}_panels.json").read_text())
    images = [Image.open(panels_dir / p["file"]).convert("RGB") for p in meta["panels"]]
    return meta, images


def draw_scale_bar(draw: ImageDraw.ImageDraw, x0: int, y_bottom: int, length_mm: float,
                   px_per_m: float, panel_w: int, font) -> None:
    """Bar of ``length_mm`` at the focus depth, anchored bottom-left of a panel."""

    if length_mm <= 0:
        return
    length_px = int(round(length_mm / 1000.0 * px_per_m))
    rule = max(3, round(panel_w * 0.008))
    pad = round(panel_w * 0.03)
    y = y_bottom - pad - 2 * rule
    x0 = x0 + pad
    x1 = x0 + length_px
    draw.line([x0, y, x1, y], fill=(40, 40, 40), width=rule)
    for x in (x0, x1):
        draw.line([x, y - 2 * rule, x, y + 2 * rule], fill=(40, 40, 40), width=rule)
    draw.text((x0, y - 2 * rule - round(font.size * 1.25)), f"{length_mm:.0f} mm",
              fill=(40, 40, 40), font=font)


def staggered_pair(top_left: Image.Image, bottom_right: Image.Image,
                   labels: tuple[str, str], font, scale: float, rule_px: int) -> Image.Image:
    """Two complete renders of the same camera in one panel, staggered.

    Each render is reduced to ``scale`` of the panel and set in its own corner,
    the first top-left and the second bottom-right, so both volumes are whole
    and as large as the footprint allows. The white ground of a render is made
    transparent before pasting, so where the two overlap the later one covers
    only with its drawn content. Each carries its label in the corner it owns.
    """

    import numpy as np

    w, h = top_left.size
    sw, sh = round(w * scale), round(h * scale)
    out = Image.new("RGB", (w, h), (255, 255, 255))

    def cutout(img: Image.Image) -> Image.Image:
        small = img.convert("RGB").resize((sw, sh), Image.LANCZOS)
        arr = np.asarray(small)
        alpha = np.where(arr.min(axis=2) < 246, 255, 0).astype(np.uint8)
        rgba = np.dstack([arr, alpha])
        return Image.fromarray(rgba, "RGBA")

    a, b = cutout(top_left), cutout(bottom_right)
    out.paste(a, (0, 0), a)
    out.paste(b, (w - sw, h - sh), b)
    draw = ImageDraw.Draw(out)
    pad = round(w * 0.025)
    halo = dict(stroke_width=max(2, rule_px), stroke_fill=(255, 255, 255))
    draw.text((pad, pad), labels[0], fill=INK, font=font, **halo)
    # The domes are rounded, so each render's outer top corner is empty: the
    # first label sits in the panel's top-left corner and the second in the
    # top-right corner of the lower render, clear of the object and its shadow.
    tw = text_w(font, labels[1])
    draw.text((w - pad - tw, h - sh + pad), labels[1], fill=INK, font=font, **halo)
    return out


def write_pdf(sheet: Image.Image, path: Path, dpi: int) -> None:
    """The sheet as a single-page PDF, Flate-compressed rather than JPEG.

    PIL's PDF writer stores RGB pages as JPEG, which smears the type; matplotlib
    stores the array losslessly, and the page size is the sheet at ``dpi``.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    w_in, h_in = sheet.width / dpi, sheet.height / dpi
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(np.asarray(sheet), interpolation="none")
    fig.savefig(path, dpi=dpi, format="pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels-dir", type=Path, required=True,
                    help="Directory written by render_nogo_volume.py --panels-dir.")
    ap.add_argument("--row", nargs="+", action="append", metavar="LETTER LABEL STEM [ALT]",
                    required=True,
                    help="One row: its letter, the object name to print, the strip "
                         "stem whose <stem>_panels.json sits in --panels-dir, and "
                         "optionally a second stem drawn in the top-left half of the "
                         "--split layer's panel.")
    ap.add_argument("--split", default=None, metavar="LAYER",
                    help="Layer whose panel holds two staggered renders: the row's "
                         "second stem top-left and its first bottom-right.")
    ap.add_argument("--split-scale", type=float, default=0.64,
                    help="Size of each staggered render as a fraction of the panel.")
    ap.add_argument("--split-labels", nargs=2, default=("k = 10", "k = 100"),
                    metavar=("TOP_LEFT", "BOTTOM_RIGHT"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale-bar-mm", type=float, default=100.0,
                    help="Scale bar on the first panel of each row; 0 to omit.")
    ap.add_argument("--legend", action=argparse.BooleanOptionalAction, default=True,
                    help="Reachable/unreachable key, right-aligned on the first row's label line.")
    ap.add_argument("--pdf", action="store_true",
                    help="Also write <out>.pdf with the raster embedded losslessly at "
                         "--dpi, for \\includegraphics.")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rows, alts = [], []
    for spec in args.row:
        if len(spec) not in (3, 4):
            raise SystemExit(f"--row takes LETTER LABEL STEM [ALT], got {spec}")
        letter, label, stem = spec[:3]
        rows.append((letter, label, *load_row(args.panels_dir, stem)))
        alts.append(load_row(args.panels_dir, spec[3]) if len(spec) == 4 else None)
    if args.split and any(a is None for a in alts):
        raise SystemExit("--split needs a second stem on every --row")
    W, H = rows[0][2]["panel_size"]
    for _, _, meta, _ in rows:
        if meta["panel_size"] != [W, H]:
            raise SystemExit("every row must share one panel size; re-render the strips "
                             "with the same --width/--height")
    names = [p["name"] for p in rows[0][2]["panels"]]
    for _, label, meta, _ in rows:
        if [p["name"] for p in meta["panels"]] != names:
            raise SystemExit(f"row {label!r} has columns {[p['name'] for p in meta['panels']]}, "
                             f"expected {names}")
    n_cols = len(names)

    # Type sizes are fractions of the panel width because the figure is reduced
    # to one column on the page; at 460 px a panel prints about 20 mm wide.
    margin = round(W * 0.035)
    gutter = round(W * 0.026)
    row_gap = round(W * 0.05)
    head_font = serif(round(W * 0.105), bold=True)
    note_font = serif(round(W * 0.074))
    letter_font = serif(round(W * 0.105), bold=True)
    label_font = serif(round(W * 0.092))
    key_font = serif(round(W * 0.074))
    bar_font = serif(round(W * 0.062))
    split_font = serif(round(W * 0.074))
    swatch = round(W * 0.058)

    key_h = 0
    head_h = round(head_font.size * 1.30)
    letter_h = round(letter_font.size * 1.25)
    note_h = round(note_font.size * 1.30)
    row_h = letter_h + note_h + H

    width = 2 * margin + n_cols * W + (n_cols - 1) * gutter
    height = (margin + key_h + head_h + len(rows) * row_h
              + (len(rows) - 1) * row_gap + margin)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    def col_x(i: int) -> int:
        return margin + i * (W + gutter)

    def draw_key(y_top: int, line_h: int) -> None:
        # Right-aligned on the first row's label line, where there is room
        # beside "A Strawberry", rather than on a line of its own. Reachable
        # first, as the panels are read: the green shell is the sampled region
        # and the red solid the part of it the strategy loses.
        colours = rows[0][2].get("colours", {})
        entries = [(colours.get("reachable", (0.145, 0.635, 0.322)), "Reachable"),
                   (colours.get("unreachable", (0.788, 0.114, 0.110)), "Unreachable")]
        gap = round(W * 0.05)
        total = sum(swatch + round(swatch * 0.45) + text_w(key_font, t) for _, t in entries) \
            + gap * (len(entries) - 1)
        x = width - margin - round(total)
        mid = y_top + line_h // 2
        for rgb, text in entries:
            draw.rectangle([x, mid - swatch // 2, x + swatch, mid + swatch // 2],
                           fill=rgb255(rgb))
            x += swatch + round(swatch * 0.45)
            draw.text((x, mid - round(key_font.size * 0.62)), text, fill=INK, font=key_font)
            x += round(text_w(key_font, text)) + gap

    y = margin
    for i, name in enumerate(names):
        draw.text((col_x(i) + (W - text_w(head_font, name)) / 2, y), name,
                  fill=INK, font=head_font)
    y += head_h

    for r, (letter, label, meta, images) in enumerate(rows):
        x = margin
        draw.text((x, y), letter, fill=INK, font=letter_font)
        x += text_w(letter_font, letter) + round(W * 0.03)
        # Baselines of the two faces align when the smaller sits lower by the
        # difference in their ascent.
        draw.text((x, y + (letter_font.size - label_font.size)), label, fill=INK,
                  font=label_font)
        if r == 0 and args.legend:
            draw_key(y, letter_h)
        y += letter_h
        alt = alts[r]
        notes = [p["note"] for p in meta["panels"]]
        images = list(images)
        if args.split and alt is not None:
            alt_meta, alt_images = alt
            for i, panel in enumerate(meta["panels"]):
                if panel["layer"] != args.split:
                    continue
                j = next(k for k, q in enumerate(alt_meta["panels"])
                         if q["layer"] == args.split)
                images[i] = staggered_pair(alt_images[j], images[i], tuple(args.split_labels),
                                           split_font, args.split_scale,
                                           max(2, round(W * 0.006)))
                # The note reads in the order the halves are seen, top-left first.
                notes[i] = (f"{alt_meta['panels'][j]['locked_fraction']:.0%} \u2192 "
                            f"{panel['locked_fraction']:.0%} unreachable")
        for i, note in enumerate(notes):
            draw.text((col_x(i) + (W - text_w(note_font, note)) / 2, y),
                      note, fill=INK, font=note_font)
        y += note_h
        for i, img in enumerate(images):
            sheet.paste(img, (col_x(i), y))
        draw_scale_bar(draw, col_x(0), y + H, args.scale_bar_mm, meta["px_per_m"], W, bar_font)
        y += H
        if r < len(rows) - 1:
            y += row_gap

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, dpi=(args.dpi, args.dpi))
    if args.pdf:
        write_pdf(sheet, args.out.with_suffix(".pdf"), args.dpi)
    for (letter, label, meta, _), alt in zip(rows, alts):
        parts = []
        for p in meta["panels"]:
            if args.split and alt is not None and p["layer"] == args.split:
                q = next(q for q in alt[0]["panels"] if q["layer"] == args.split)
                parts.append(f"{p['name']} {q['locked_fraction']:.0%} -> "
                             f"{p['locked_fraction']:.0%} unreachable")
            else:
                parts.append(f"{p['name']} {p['note']}")
        print(f"  {letter} {label}: {', '.join(parts)}")
    print(f"wrote {args.out}  ({width}x{height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
