#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

DEFAULT_ROOTS = ["resource", "resources"]
GIF_WIDTH = 720
GIF_FPS = 10
MP4_CRF = 23
X264_PRESET = "veryfast"
AUDIO_BR = "128k"

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise SystemExit("Error: ffmpeg not found. Install it first (e.g., `brew install ffmpeg`).")

def is_up_to_date(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    return dst.stat().st_mtime >= src.stat().st_mtime

def run_ffmpeg(cmd):
    # Show a concise command; capture errors nicely
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Print stderr for debugging
        print(f"[ffmpeg error] {' '.join(cmd)}")
        print(e.stderr.decode(errors="ignore"))
        raise

def convert_one(src: Path, crf: int, preset: str, gif_width: int, gif_fps: int, audio_br: str, dry_run: bool):
    base = src.with_suffix("")  # drop extension
    mp4 = base.with_suffix(".mp4")
    gif = base.with_suffix(".gif")

    # MP4
    if not is_up_to_date(src, mp4):
        print(f"→ MOV -> MP4: {src} -> {mp4} (crf={crf}, preset={preset})")
        if not dry_run:
            cmd_mp4 = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", audio_br,
                str(mp4),
            ]
            run_ffmpeg(cmd_mp4)
    else:
        print(f"✓ MP4 up-to-date: {mp4}")

    # GIF
    if not is_up_to_date(src, gif):
        print(f"→ MOV -> GIF: {src} -> {gif} (fps={gif_fps}, width={gif_width})")
        if not dry_run:
            cmd_gif = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-vf", f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos",
                "-loop", "0",
                str(gif),
            ]
            run_ffmpeg(cmd_gif)
    else:
        print(f"✓ GIF up-to-date: {gif}")

    return mp4, gif

def scan_and_convert(roots, crf, preset, gif_width, gif_fps, audio_br, dry_run):
    found = []
    for root in roots:
        d = Path(root)
        if not d.exists():
            continue
        for src in d.rglob("*.mov"):
            mp4, gif = convert_one(src, crf, preset, gif_width, gif_fps, audio_br, dry_run)
            found.append((src, mp4, gif))
    if not found:
        print("No .mov files found.")
    return found

def make_readme_markdown(pairs, raw_base=None):
    """
    pairs: list of tuples (src_mov, mp4_path, gif_path)
    raw_base: optional base for MP4 raw links, e.g.:
              'https://github.com/correlllab/rh56_controller/raw/main'
              If None, use relative links.
    """
    lines = []
    lines.append("## Demos\n")
    lines.append(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}._\n")
    lines.append("| Preview | Video |\n|:--|:--|\n")
    for _, mp4, gif in pairs:
        gif_rel = gif.as_posix()
        mp4_link = mp4.as_posix() if raw_base is None else f"{raw_base}/{mp4.as_posix()}"
        # GIF preview clicking to MP4
        lines.append(f"| ![]({gif_rel}) | [▶︎ View MP4]({mp4_link}) |\n")
    return "".join(lines)

def parse_args():
    p = argparse.ArgumentParser(description="Batch transcode .mov -> .mp4 and .gif, and emit README markdown.")
    p.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Directories to scan (default: resource resources)")
    p.add_argument("--crf", type=int, default=MP4_CRF, help="x264 CRF (18–28; lower=better). Default: 23")
    p.add_argument("--preset", default=X264_PRESET, help="x264 preset (ultrafast..veryslow). Default: veryfast")
    p.add_argument("--gif-width", type=int, default=GIF_WIDTH, help="GIF width in pixels (height auto). Default: 720")
    p.add_argument("--gif-fps", type=int, default=GIF_FPS, help="GIF frames per second. Default: 10")
    p.add_argument("--audio-br", default=AUDIO_BR, help="Audio bitrate for MP4 (e.g., 128k).")
    p.add_argument("--dry-run", action="store_true", help="Print actions without converting.")
    p.add_argument("--emit-markdown", action="store_true", help="Print README markdown for the converted files.")
    p.add_argument("--raw-base", default=None,
                   help="Optional base URL for raw MP4 links (e.g., https://github.com/<org>/<repo>/raw/main)")
    return p.parse_args()

def main():
    args = parse_args()
    check_ffmpeg()
    pairs = scan_and_convert(
        roots=args.roots,
        crf=args.crf,
        preset=args.preset,
        gif_width=args.gif_width,
        gif_fps=args.gif_fps,
        audio_br=args.audio_br,
        dry_run=args.dry_run
    )
    if args.emit_markdown and pairs:
        print()
        print(make_readme_markdown(pairs, raw_base=args.raw_base))

if __name__ == "__main__":
    main()
