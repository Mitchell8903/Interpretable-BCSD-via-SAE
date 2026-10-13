#!/usr/bin/env bash
# Download the jTrans checkpoints, released embeddings, and the BinaryCorp-3M
# test split.

set -euo pipefail

cd "$(dirname "$0")/.."

BASE=https://cloud.vul337.team:8443

# Expected sizes in bytes
SMALL_TEST_BYTES=780855782
EXPERIMENTS_BYTES=441882541
MODELS_BYTES=1294853487

# url dest expected_bytes [extra curl args...]
download() {
    local url=$1 dest=$2 expected=$3
    shift 3
    if [ -e "$dest" ] && [ "$(stat -c %s "$dest")" = "$expected" ]; then
        echo "have $dest"
        return
    fi
    echo "downloading $dest ($expected bytes)"
    curl -# -C - --retry 3 -L "$@" -o "$dest" "$url"
    local actual
    actual=$(stat -c %s "$dest")
    [ "$actual" = "$expected" ] || { echo "$dest: got $actual bytes, expected $expected" >&2; exit 1; }
}

extract() {
    local tarball=$1 inner=$2 dest=$3
    local tmp=${dest}.partial
    rm -rf "$tmp"
    mkdir -p "$tmp"
    tar xzf "$tarball" -C "$tmp"
    mkdir -p "$(dirname "$dest")"
    mv "$tmp/$inner" "$dest"
    rm -rf "$tmp"
}

mkdir -p data

# BinaryCorp is a folder share, so the file is fetched over WebDAV with the
# share token as the username. The others are single-file shares.
if [ -d data/BinaryCorp/small_test ]; then
    echo "have data/BinaryCorp/small_test"
else
    download "$BASE/public.php/webdav/small_test.tar.gz" \
             data/small_test.tar.gz "$SMALL_TEST_BYTES" \
             --user "cxnH8DfZTADLKCs:"
    extract data/small_test.tar.gz small_test data/BinaryCorp/small_test
fi

if [ -d experiments/BinaryCorp-3M ]; then
    echo "have experiments/BinaryCorp-3M"
else
    download "$BASE/s/wmqzYFyJnSEfEgm/download" data/experiments.tar.gz "$EXPERIMENTS_BYTES"
    extract data/experiments.tar.gz experiments/BinaryCorp-3M experiments/BinaryCorp-3M
fi

if [ -d models/jTrans-finetune ]; then
    echo "have models/jTrans-finetune"
else
    download "$BASE/s/tM5qGQPJa6iynCf/download" data/models.tar.gz "$MODELS_BYTES"
    extract data/models.tar.gz models models
fi

echo "done"
