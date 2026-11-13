"""Speak the story.

macOS ships `say`, so this needs no package and works offline. On other
platforms it degrades to doing nothing rather than crashing the app.

    from tts import speak
    speak("Good morning, teacher.")
"""

import platform
import shutil
import subprocess
import threading


def available():
    return platform.system() == "Darwin" and shutil.which("say") is not None


def speak(text, voice=None, rate=None, blocking=False):
    """Speak text aloud. Returns True if speech was started.

    Non-blocking by default so a Streamlit callback does not freeze the UI
    for the length of the sentence.
    """
    if not text or not available():
        return False

    cmd = ["say"]
    if voice:
        cmd += ["-v", voice]
    if rate:
        cmd += ["-r", str(rate)]
    cmd.append(text)

    def run():
        try:
            subprocess.run(cmd, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    if blocking:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()
    return True


def to_file(text, path, voice=None):
    """Write speech to an AIFF file, for embedding in a report or slide."""
    if not text or not available():
        return False
    cmd = ["say", "-o", str(path)]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    return subprocess.run(cmd, check=False).returncode == 0
