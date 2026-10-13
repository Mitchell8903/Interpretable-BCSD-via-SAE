#!/usr/bin/env bash
# Download the trained SAE checkpoints from the GitHub release

set -euo pipefail

cd "$(dirname "$0")/.."

BASE=https://github.com/Mitchell8903/Interpretable-BCSD-via-SAE/releases/download/checkpoints-v1
MANIFEST=results/checkpoints.sha256

# Each manifest line is "<sha256>  <path to sae.pt>"
while read -r sum path; do
    run=$(basename "$(dirname "$path")")
    if [ -e "$path" ] && echo "$sum  $path" | sha256sum -c --quiet - 2>/dev/null; then
        echo "have $path"
        continue
    fi
    echo "downloading $path"
    mkdir -p "$(dirname "$path")"
    curl -# --retry 3 -L -o "$path" "$BASE/$run.pt"
    echo "$sum  $path" | sha256sum -c --quiet - || { echo "$path: checksum mismatch" >&2; exit 1; }
done < "$MANIFEST"

echo "done"
