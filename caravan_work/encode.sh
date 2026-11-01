#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: bash caravan_work/encode.sh input.wav output.mp3" >&2
  exit 2
fi
ffmpeg -n -i "$1" -map 0:a:0 -vn -map_metadata -1 -c:a libmp3lame -b:a 320k "$2"
