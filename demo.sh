#!/usr/bin/env bash
# Eyewitness — one-command live demo.
#   ./demo.sh path/to/clip.mp4
# Runs your video through the full pipeline, writes the evidence trail to
# Butterbase, and opens the live dashboard so you watch the case appear.

set -euo pipefail

CLIP="${1:-clip.mp4}"
PY="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
[ -x "$PY" ] || PY="$(command -v python3)"

cd "$(dirname "$0")"

if [ ! -f "$CLIP" ]; then
  echo "Usage: ./demo.sh path/to/clip.mp4"
  echo "No video found at: $CLIP"
  echo "Download a dashcam near-miss clip and pass its path."
  exit 1
fi

exec "$PY" scripts/live_demo.py --clip "$CLIP"
