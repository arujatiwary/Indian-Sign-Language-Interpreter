# Real-Time Sign Language Storyteller

Phase 1 and 2 of the build: landmark extraction, cached dataset, and the sign
classifier. Real-time capture, the GenAI layer, and the Streamlit app come next.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh
python -m pytest tests -q
```

## Pipeline

```bash
# 1. sanity check the camera path before the slow bulk run
python src/check_webcam.py

# 2. see what signs your INCLUDE download actually contains
python src/extract_dataset.py --include-dir ~/Datasets/INCLUDE --list-only

# 3. pick 20-50 of those into configs/vocab.txt, one per line, then extract
python src/extract_dataset.py --include-dir ~/Datasets/INCLUDE \
    --vocab configs/vocab.txt --out data

# 4. train
python src/train.py --data data --epochs 60
```

Extraction is the slow step. Budget roughly the length of the videos
themselves. It runs once and caches to `data/seqs/*.npy`; training after that
takes seconds.

## Four places this differs from the spec

**Two hands and body pose, not one hand.** The spec says 21 landmarks and 63
features. That is one hand. ISL is heavily two-handed, and sign location
relative to the body is meaning-bearing: the same handshape at the chest and at
the forehead are different signs. One hand alone throws that away. The feature
vector is 282 floats per frame, doubled to 564 by frame-to-frame deltas.

**MediaPipe Tasks API, not `mp.solutions`.** mediapipe removed `mp.solutions`
in the 0.10.3x releases. Every tutorial you find using `mp.solutions.hands`
fails at import on a current install. The Tasks API needs the `.task` bundles
on disk, which is what `download_models.sh` fetches.

**Features are normalised to the body frame.** Every landmark is expressed
relative to the shoulder midpoint and divided by shoulder width. Without this,
a model trained on INCLUDE video sees your webcam framing as a different
distribution and the accuracy falls off a cliff. Handshape gets a second
wrist-local copy so it stays separate from where the arm is.

**Standardisation uses fixed training-set statistics, not LayerNorm.**
LayerNorm rescales each frame on its own and would undo the body
normalisation. The mean and std are buffers in the checkpoint, so the webcam
path normalises identically.

## Two things to watch

**Handedness flips when you mirror.** MediaPipe labels hands from the image's
point of view. Webcam previews are usually mirrored for the user. Mirror the
display only. Feed the unmirrored frame to the extractor, or left and right
swap against the training data.

**Vocabulary is constrained by INCLUDE, not by your story.** The example story
in the spec uses park, football, and friend. Check with `--list-only` which of
those exist. Pick your demo sentences from the labels that are actually in the
dataset, otherwise the story cannot be signed.

## Layout

```
src/config.py           feature layout and hyperparameters, single source of truth
src/features.py         normalisation maths, pure numpy, unit tested
src/extractor.py        mediapipe wrapper, video and live frames
src/check_webcam.py     sanity check on webcam or one video
src/extract_dataset.py  INCLUDE videos -> cached .npy
src/dataset.py          stratified split and augmentation
src/model.py            BiLSTM classifier
src/train.py            training, metrics, confusion matrix
```

## Next

1. `src/realtime.py`: rolling buffer, confidence gate, duplicate collapse.
2. Phrase boundary detection, since nothing yet decides when a sign sequence is
   a complete phrase.
3. GenAI prompt with the running story as context and a hard instruction
   against inventing events.
4. Streamlit app.
