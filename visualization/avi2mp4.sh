#!/usr/bin/env bash
#
# avi2mp4.sh - Convert an .avi to a LinkedIn-friendly .mp4.
#
# * Keeps EVERY input frame (nothing is dropped).
# * If the clip is shorter than MIN_DURATION seconds, it is slowed down
#   (each frame is held longer) so the result is longer than LinkedIn's
#   3-second minimum.
# * H.264 + yuv420p + faststart, which LinkedIn and all major players accept.
#
# Usage:
#   ./avi2mp4.sh input.avi [output.mp4]
#
# Optional environment variables:
#   MIN_DURATION  minimum output length in seconds (default: 4)
#   OUT_FPS       output frame rate              (default: 30)
#   CRF           x264 quality, lower = better   (default: 18)
#
# Install ffmpeg on Ubuntu with:  sudo apt install ffmpeg

set -euo pipefail

MIN_DURATION="${MIN_DURATION:-4}"
OUT_FPS="${OUT_FPS:-30}"
CRF="${CRF:-18}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 input.avi [output.mp4]" >&2
    exit 1
fi

in="$1"
out="${2:-${in%.*}.mp4}"

for cmd in ffmpeg ffprobe awk; do
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "Error: '$cmd' not found. Install with: sudo apt install ffmpeg" >&2
        exit 1
    }
done

[[ -f "$in" ]] || { echo "Error: input file '$in' not found." >&2; exit 1; }

# Count the frames actually present in the file (more reliable than the header).
frames=$(ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$in")
# Source frame rate, e.g. "20/1"
rate=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=r_frame_rate -of default=nw=1:nk=1 "$in")

if [[ -z "$frames" || "$frames" == "N/A" || "$frames" -lt 1 ]]; then
    echo "Error: could not read the video stream of '$in'." >&2
    exit 1
fi

# Original duration, and the slow-down factor needed to reach MIN_DURATION.
read -r orig_dur factor target_dur < <(awk -v n="$frames" -v r="$rate" -v m="$MIN_DURATION" '
    BEGIN {
        split(r, a, "/"); fps = (a[2] > 0) ? a[1] / a[2] : a[1]
        d = n / fps
        f = (d < m) ? m / d : 1
        printf "%.6f %.6f %.6f\n", d, f, d * f
    }')

echo "Input:    $in"
echo "Frames:   $frames @ $rate fps  (${orig_dur}s)"
echo "Output:   $out"
echo "Slowdown: x${factor}  ->  ${target_dur}s at ${OUT_FPS} fps"

# Filter chain:
#   setpts : stretch timestamps (every frame is kept, just shown for longer)
#   fps    : resample to a standard constant frame rate (repeats frames, never drops)
#   scale  : force even width/height (required by H.264 + yuv420p)
#   format : yuv420p for maximum compatibility (input is full-range yuvj420p)
ffmpeg -hide_banner -loglevel error -stats -y -i "$in" \
    -an \
    -vf "setpts=${factor}*PTS,fps=${OUT_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
    -c:v libx264 -preset slow -crf "$CRF" \
    -movflags +faststart \
    "$out"

# Report what we produced.
out_frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration \
    -of default=nw=1:nk=1 "$out")
echo "Done. Output duration: ${out_frames}s"
