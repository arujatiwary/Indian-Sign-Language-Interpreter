"""Streamlit front end: camera or demo video, detected signs, story, speech.

    streamlit run app.py

Layout follows the wireframe in the project spec: video on the left, detected
signs on the right, story across the bottom, Speak and Clear underneath.

Two things about Streamlit shape this file.

Streamlit reruns the whole script on every interaction, so anything that must
survive a rerun lives in st.session_state. That includes the open camera, the
rolling buffer and the phrase detector, all of which are stateful.

Streamlit also cannot process a click while a script is running. An earlier
version captured frames in `while st.session_state.running:`, which meant the
Stop button could never be read and the only way out was killing the app. The
loop here takes a fixed batch of frames and then calls st.rerun(), so clicks
land between batches.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import CKPT_DIR  # noqa: E402
from phrase import PhraseConfig, PhraseDetector  # noqa: E402
from realtime import RollingBuffer, SignRecognizer  # noqa: E402

# Anchored to this file, not the working directory: `streamlit run app.py`
# can be launched from anywhere and a relative path would miss.
ROOT = Path(__file__).resolve().parent
DEMO_VIDEO = ROOT / "demo" / "story.mp4"
DEMO_RESULT = ROOT / "demo" / "story.result.json"
BATCH = 20          # frames per rerun. Each rerun re-executes the whole
                    # script, so fewer is smoother, but Stop can only be read
                    # between batches. 20 frames is under a second either way.
PREVIEW_W = 960     # frames are 1920 wide; shipping that to the browser is
                    # the slow part of the loop and buys nothing on screen
FONT = cv2.FONT_HERSHEY_SIMPLEX

st.set_page_config(page_title="Sign Storyteller", layout="wide")


# ---- cached heavy objects -------------------------------------------------

@st.cache_resource
def load_recognizer(checkpoint):
    """Loading the checkpoint on every rerun would make the UI crawl."""
    return SignRecognizer(checkpoint)


@st.cache_resource
def load_extractor():
    from extractor import LandmarkExtractor
    return LandmarkExtractor(running_mode="video")


# ---- session state --------------------------------------------------------

def fresh_run_state():
    return {
        "phrases": [], "sentences": [], "paragraph": None,
        "frame_idx": 0, "fps": 0.0, "last_label": None, "last_conf": 0.0,
        "last_frame": None,
        "detector": PhraseDetector(PhraseConfig()),
        "buffer": RollingBuffer(),
    }


def init_state():
    base = {"running": False, "capture": None, "source": "Webcam",
            "teller": None, "error": None}
    base["last_frame"] = None
    base.update(fresh_run_state())
    for k, v in base.items():
        st.session_state.setdefault(k, v)


def release_capture():
    if st.session_state.capture is not None:
        st.session_state.capture.release()
        st.session_state.capture = None


def reset_run():
    release_capture()
    if st.session_state.get("teller") is not None:
        st.session_state.teller.clear()
    for k, v in fresh_run_state().items():
        st.session_state[k] = v
    st.session_state.error = None


init_state()


# ---- sidebar --------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    source = st.radio("Source", ["Webcam", "Demo video"],
                      index=0 if st.session_state.source == "Webcam" else 1,
                      disabled=st.session_state.running,
                      help="The demo video runs the same pipeline on stitched "
                           "dataset clips, so the system can be shown without "
                           "a camera.")
    if source != st.session_state.source:
        st.session_state.source = source
        reset_run()

    ckpts = sorted(p.name for p in CKPT_DIR.glob("*.pt")) or ["best.pt"]
    checkpoint = st.selectbox(
        "Model", ckpts,
        index=ckpts.index("best.pt") if "best.pt" in ckpts else 0,
        disabled=st.session_state.running)
    camera = st.number_input("Camera index", 0, 4, 0,
                             disabled=st.session_state.running
                             or source != "Webcam")
    confidence = st.slider("Confidence threshold", 0.3, 0.95, 0.75, 0.05,
                           help="A prediction below this is discarded. 0.75 "
                                "is what rejects the spurious signs seen "
                                "during pauses.")
    predict_every = st.slider("Predict every N frames", 1, 6, 3)

    st.divider()
    use_story = st.checkbox("Generate language", value=True)
    paragraph_mode = st.checkbox(
        "Paragraph mode", value=True,
        help="One connected paragraph over all phrases, rather than one "
             "sentence per phrase. This is the mode the project describes.")
    offline = st.checkbox(
        "Offline (replay saved paragraph)", value=False,
        help="Use the paragraph saved from an earlier run instead of calling "
             "the API. Everything else still runs for real.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1,
                            help="Fixed and recorded, so ratings are "
                                 "reproducible")
    st.caption("The preview is mirrored. The extractor always sees the "
               "unmirrored frame, or left and right swap against training.")


# ---- header ---------------------------------------------------------------

st.title("Real-Time Sign Language Storyteller")

try:
    recog = load_recognizer(str(CKPT_DIR / checkpoint))
except SystemExit as e:
    st.error(str(e))
    st.stop()

if use_story and st.session_state.teller is None:
    from storyteller import Storyteller
    st.session_state.teller = Storyteller(temperature=temperature)
teller = st.session_state.teller if use_story else None
if teller:
    # The slider has to reach the model. The Storyteller is cached across
    # reruns, so setting this only at construction froze it at the first value.
    teller.temperature = temperature

bits = [f"{len(recog.labels)} signs", str(recog.device)]
if teller:
    bits.append(f"{teller.provider}"
                + (f" / {teller.model_used}" if teller.model_used else ""))
st.caption(" · ".join(bits))

if teller and teller.provider == "none" and not offline:
    st.warning("No API key found. Sentences will use the offline template, "
               "which is not model output. Put a key in .env, or tick "
               "Offline to replay a saved paragraph.")
if st.session_state.source == "Demo video" and not DEMO_VIDEO.exists():
    st.error(f"{DEMO_VIDEO} not found. Build it with "
             "`python src/demo_video.py --build`.")


# ---- controls -------------------------------------------------------------

c1, c2, c3, c4, _ = st.columns([1, 1, 1.2, 1.4, 3])

if c1.button("Start", type="primary", disabled=st.session_state.running):
    reset_run()
    st.session_state.running = True
    st.rerun()

if c2.button("Stop", disabled=not st.session_state.running):
    st.session_state.running = False
    release_capture()
    st.rerun()

if c3.button("End phrase", disabled=not st.session_state.running):
    phrase = st.session_state.detector.flush()
    if phrase:
        st.session_state.phrases.append(phrase)

if c4.button("Tell the story",
             disabled=not st.session_state.phrases or not use_story):
    st.session_state.paragraph = None
    with st.spinner("Generating..."):
        if offline:
            import json
            if DEMO_RESULT.exists():
                rec = json.loads(DEMO_RESULT.read_text())
                st.session_state.paragraph = rec.get("paragraph")
                st.session_state.error = (
                    None if rec.get("phrases") == st.session_state.phrases
                    else "Replayed paragraph: it was written for different "
                         "phrases than the ones just detected.")
            else:
                st.session_state.error = f"No saved result at {DEMO_RESULT}."
        elif paragraph_mode:
            st.session_state.paragraph = teller.narrate(
                st.session_state.phrases)
            st.session_state.error = teller.last_error
        else:
            # add() appends to the teller's own running story, which is the
            # context it sends with each call. Without clearing, a second
            # click would generate against the first click's output.
            teller.clear()
            st.session_state.sentences = [teller.add(p)
                                          for p in st.session_state.phrases]
            st.session_state.error = teller.last_error


# ---- panels ---------------------------------------------------------------

left, right = st.columns([3, 1])
with left:
    video_slot = st.empty()
    # Painted immediately, not left blank. st.empty() is recreated on every
    # rerun, and with a rerun every batch the panel spent the gap between
    # batches showing nothing, which read as heavy flicker.
    if st.session_state.last_frame is not None:
        video_slot.image(st.session_state.last_frame, channels="RGB",
                         width="stretch")
with right:
    st.subheader("Detected signs")
    signs_slot = st.empty()
    stats_slot = st.empty()

st.subheader("Story")
story_slot = st.empty()
note_slot = st.empty()

b1, b2, _ = st.columns([1, 1, 4])
speak_clicked = b1.button("Speak story")
if b2.button("Clear story"):
    reset_run()
    st.rerun()


def story_text():
    if st.session_state.paragraph:
        return st.session_state.paragraph
    return " ".join(s for s in st.session_state.sentences if s)


def render_panels():
    pending = st.session_state.detector.words
    lines = []
    for i, phrase in enumerate(st.session_state.phrases, 1):
        lines.append(f"**{i}.** {' '.join(w.upper() for w in phrase)}")
    if pending:
        lines.append("_" + " ".join(w.upper() for w in pending) + "_")
    signs_slot.markdown("\n\n".join(lines) or "_nothing yet_")

    label = st.session_state.last_label
    conf = st.session_state.last_conf
    stats_slot.caption(
        f"frame {st.session_state.frame_idx} · {st.session_state.fps:.0f} fps"
        + (f"\n\n{label} {conf:.2f}" if label else ""))
    story_slot.info(story_text() or "_no story yet_")
    if st.session_state.error:
        note_slot.warning(st.session_state.error)
    else:
        note_slot.empty()


render_panels()

if speak_clicked and story_text():
    from tts import available, speak
    if available():
        speak(story_text())
    else:
        st.warning("Text to speech needs macOS `say`.")


# ---- capture --------------------------------------------------------------

def open_source():
    """Open the camera or the demo file, once, and keep it across reruns."""
    if st.session_state.source == "Demo video":
        cap = cv2.VideoCapture(str(DEMO_VIDEO))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {DEMO_VIDEO}")
        return cap
    from realtime import _open_camera
    return _open_camera(cv2, int(camera))


def draw(frame, label, conf, mirror):
    view = cv2.flip(frame, 1) if mirror else frame.copy()
    if view.shape[1] > PREVIEW_W:
        scale = PREVIEW_W / view.shape[1]
        view = cv2.resize(view, (PREVIEW_W, int(view.shape[0] * scale)))
    h, w = view.shape[:2]
    band = view[0:56, 0:w]
    view[0:56, 0:w] = cv2.addWeighted(band, 0.4, np.zeros_like(band), 0.6, 0)
    if label and conf >= confidence:
        text, colour = f"{label.upper()}  {conf:.2f}", (120, 230, 120)
    elif label:
        text, colour = (f"{label} {conf:.2f} (below {confidence:.2f})",
                        (235, 170, 120))
    else:
        text, colour = "listening", (170, 170, 170)
    cv2.putText(view, text, (18, 38), FONT, 0.85, colour, 2, cv2.LINE_AA)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


if st.session_state.running:
    if st.session_state.capture is None:
        try:
            st.session_state.capture = open_source()
        except SystemExit as e:
            st.session_state.running = False
            st.session_state.error = str(e)
            st.rerun()

    cap = st.session_state.capture
    ext = load_extractor()
    buf, det = st.session_state.buffer, st.session_state.detector
    det.cfg.min_confidence = confidence
    is_video = st.session_state.source == "Demo video"

    # A video file has no clock of its own, so without pacing the loop runs
    # as fast as the machine can decode: the first run went through 555
    # frames at 107 fps and the whole thing was over in five seconds. A
    # camera paces itself, so this applies to the file only.
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    budget = 1.0 / src_fps if is_video else 0.0

    t0, shown, ended = time.perf_counter(), None, False
    for _ in range(BATCH):
        tick = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            ended = True
            break

        i = st.session_state.frame_idx
        # Video replays at a fixed rate; the camera does not, so its timestamps
        # come from the clock. The extractor enforces monotonicity either way.
        ts = int(i * 40) if is_video else int(time.time() * 1000)
        buf.push(ext.frame(frame, ts))
        st.session_state.frame_idx = i + 1

        label, conf = None, 0.0
        if buf.ready and st.session_state.frame_idx % predict_every == 0:
            label, conf = recog.predict(buf.sequence(recog.use_motion))
            st.session_state.last_label = label
            st.session_state.last_conf = conf

        phrase = det.update(label, conf, buf.hands_present_recent())
        if phrase:
            st.session_state.phrases.append(phrase)
        # Drawn inside the loop, not after it: showing only the last frame of
        # each batch is a slideshow, and st.empty() does push updates while
        # the script is still running.
        shown = draw(frame, label, conf, mirror=not is_video)
        video_slot.image(shown, channels="RGB", width="stretch")

        spare = budget - (time.perf_counter() - tick)
        if spare > 0:
            time.sleep(spare)

    if shown is not None:
        st.session_state.last_frame = shown
    dt = time.perf_counter() - t0
    st.session_state.fps = BATCH / dt if dt > 0 else 0.0
    render_panels()

    if ended:
        tail = det.flush()
        if tail:
            st.session_state.phrases.append(tail)
        st.session_state.running = False
        release_capture()
        st.rerun()
    else:
        st.rerun()
else:
    # Keep the last frame on screen. Replacing it with a message makes the
    # panel look like nothing ever played.
    if st.session_state.last_frame is None:
        video_slot.info("Press **Start**.")
    elif st.session_state.phrases:
        note_slot.success("Finished. Press **Tell the story**.")
