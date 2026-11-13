#!/usr/bin/env bash
# Fetch the mediapipe Tasks model bundles. Run once from the project root.
set -euo pipefail
mkdir -p models
BASE="https://storage.googleapis.com/mediapipe-models"
curl -L -o models/hand_landmarker.task \
  "$BASE/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
curl -L -o models/pose_landmarker_lite.task \
  "$BASE/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
ls -lh models
