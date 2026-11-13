"""Patch src/check_webcam.py in place to report frame brightness.

Run once from the project root:

    python3 patch_check_webcam.py

Safe to run twice; it detects an already-patched file and does nothing.
"""
from pathlib import Path

TARGET = Path("src/check_webcam.py")

OLD_HINT = '''    "enable your terminal (Terminal, iTerm, or VS Code), then fully quit and "
    "reopen it. the prompt only appears once, so if you dismissed it you must "
    "enable it by hand."
)'''

NEW_HINT = '''    "enable your terminal (Terminal, iTerm, or VS Code), then fully quit it "
    "with Cmd-Q and reopen. a running process keeps its old permission state.\\n"
    "note that a denied permission does not raise an error on macOS. the "
    "camera opens and hands back all-black frames, so 'it opened fine' is not "
    "evidence that it works."
)

BLACK_THRESHOLD = 6.0   # mean pixel value below this is effectively a black frame'''

OLD_PRINT = '''            print(f"  {idx}: OK, {w}x{h}")'''

NEW_PRINT = '''            brightness = float(frame.mean())
            if brightness < BLACK_THRESHOLD:
                print(f"  {idx}: opened, {w}x{h}, but frames are BLACK "
                      f"(mean {brightness:.1f}) -- almost always denied permission")
            else:
                print(f"  {idx}: OK, {w}x{h}, mean {brightness:.1f}")'''

OLD_WARMUP = '''        for attempt in range(60):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(0.05)'''

NEW_WARMUP = '''        first = None
        for attempt in range(60):
            ok, first = cap.read()
            if ok:
                break
            time.sleep(0.05)'''

ANCHOR = '''    vectors = []'''

GUARD = '''        if first is None or float(first.mean()) < BLACK_THRESHOLD:
            cap.release()
            raise SystemExit(
                f"camera {args.camera} is delivering black frames.\\n"
                f"{PERMISSION_HINT}"
            )

    vectors = []'''


def main():
    if not TARGET.exists():
        raise SystemExit(f"{TARGET} not found. run this from the project root.")

    text = TARGET.read_text()
    if "BLACK_THRESHOLD" in text:
        print("already patched, nothing to do")
        return

    edits = 0
    for old, new, name in [
        (OLD_HINT, NEW_HINT, "brightness constant"),
        (OLD_PRINT, NEW_PRINT, "probe output"),
        (OLD_WARMUP, NEW_WARMUP, "warm-up capture"),
    ]:
        if old not in text:
            raise SystemExit(f"could not find the '{name}' section. file differs from expected.")
        text = text.replace(old, new, 1)
        edits += 1

    if ANCHOR not in text:
        raise SystemExit("could not find the insertion point for the black-frame guard.")
    text = text.replace(ANCHOR, GUARD, 1)
    edits += 1

    TARGET.write_text(text)
    print(f"applied {edits} edits to {TARGET}")
    print("verify with: grep -c BLACK_THRESHOLD src/check_webcam.py   (expect 3)")


if __name__ == "__main__":
    main()
