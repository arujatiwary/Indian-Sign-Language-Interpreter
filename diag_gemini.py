"""Isolate the Gemini call from the rest of the project.

Run from the project root:

    python3 diag_gemini.py

Prints exactly what the request looks like and what comes back, so the failure
is visible instead of being swallowed by the try/except in Storyteller.add().
"""
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

# 1. what the module thinks it is doing
import storyteller as st
print("module file :", st.__file__)
print("provider    :", st._detect_provider())
print("model       :", st.DEFAULT_MODELS.get("gemini"))
print("key set     :", bool(os.environ.get("GEMINI_API_KEY")))
print("other keys  :", [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
                        if os.environ.get(k)])
print()

# 2. the raw call, with the error surfaced
key = os.environ.get("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY is not set in this shell")

model = st.DEFAULT_MODELS["gemini"]
url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
       f"{model}:generateContent?key={key}")
payload = {
    "system_instruction": {"parts": [{"text": "Reply with one short sentence."}]},
    "contents": [{"parts": [{"text": "Glosses: I HAPPY\nSentence:"}]}],
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
}
ctx = ssl.create_default_context(cafile=__import__("certifi").where())
req = urllib.request.Request(
    url, data=json.dumps(payload).encode(),
    headers={"content-type": "application/json"}, method="POST")

try:
    with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
        data = json.load(r)
    print("RAW RESPONSE:")
    print(json.dumps(data, indent=2)[:1500])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}")
    print(e.read().decode()[:1500])
except Exception as e:
    print("FAILED:", repr(e))

# 3. the same call through the module
print("\nvia Storyteller:")
s = st.Storyteller()
print("  sentence:", s.add(["i", "happy"]))
print("  error   :", s.last_error)