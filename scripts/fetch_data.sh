#!/usr/bin/env bash
# COCO 2014 val captions (40,504 images x 5 captions): encoder training and the COCO benchmark.
set -euo pipefail
mkdir -p data
[ -f data/captions_val2014.json ] || curl -fL -o data/captions_val2014.json \
  https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/captions_val2014.json
