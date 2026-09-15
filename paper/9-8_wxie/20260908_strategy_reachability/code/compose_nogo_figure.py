#!/usr/bin/env python3
"""Lay out the no-go strips of several objects as one lettered grid.

render_nogo_volume.py --strip --panels-dir writes each strip's panels bare and a
<stem>_panels.json sidecar naming them. This script stacks those strips into the
paper figure: one column per strategy with the strategy named once at the top, a
lettered row per object, the unreachable fraction over each panel, a shared
reachable/unreachable key, and a scale bar on the first panel of every row. Type
is set in the paper's serif face so the figure reads as part of the page.

Usage:
    uv run python compose_nogo_figure.py --panels-dir ../figures/panels \\
        --row A Strawberry 20260908-nogo_strip_strawberry_ggx \\
        --row B Can 20260908-nogo_strip_can_ggx \\
        --out ../figures/20260908-nogo_grid_strawberry_can.png
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
    ap.add_argument("--row", nargs=3, action="append", metavar=("LETTER", "LABEL", "STEM"),
                    required=True,
                    help="One row: its letter, the object name to print, and the "
                         "strip stem whose <stem>_panels.json sits in --panels-dir.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale-bar-mm", type=float, default=100.0,
                    help="Scale bar on the first panel of each row; 0 to omit.")
    ap.add_argument("--legend", action=argparse.BooleanOptionalAction, default=True,
                    help="Reachable/unreachable key, top right.")
    ap.add_argument("--pdf", action="store_true",
                    help="Also write <out>.pdf with the raster embedded losslessly at "
                         "--dpi, for \\includegraphics.")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    rows = [(letter, label, *load_row(args.panels_dir, stem))
            for letter, label, stem in args.row]
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
    swatch = round(W * 0.058)

    key_h = round(key_font.size * 1.35) if args.legend else 0
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

    y = margin
    if args.legend:
        # Reachable first, as the panels are read: the green shell is the sampled
        # region and the red solid the part of it the strategy loses.
        colours = rows[0][2].get("colours", {})
        entries = [(colours.get("reachable", (0.145, 0.635, 0.322)), "Reachable"),
                   (colours.get("unreachable", (0.788, 0.114, 0.110)), "Unreachable")]
        gap = round(W * 0.05)
        total = sum(swatch + round(swatch * 0.45) + text_w(key_font, t) for _, t in entries) \
            + gap * (len(entries) - 1)
        x = width - margin - round(total)
        mid = y + key_h // 2
        for rgb, text in entries:
            draw.rectangle([x, mid - swatch // 2, x + swatch, mid + swatch // 2],
                           fill=rgb255(rgb))
            x += swatch + round(swatch * 0.45)
            draw.text((x, mid - round(key_font.size * 0.62)), text, fill=INK, font=key_font)
            x += round(text_w(key_font, text)) + gap
        y += key_h

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
        y += letter_h
        for i, panel in enumerate(meta["panels"]):
            draw.text((col_x(i) + (W - text_w(note_font, panel["note"])) / 2, y),
                      panel["note"], fill=INK, font=note_font)
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
    for letter, label, meta, _ in rows:
        notes = ", ".join(f"{p['name']} {p['note']}" for p in meta["panels"])
        print(f"  {letter} {label}: {notes}")
    print(f"wrote {args.out}  ({width}x{height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
