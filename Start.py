"""
SignalScope v4.1 — Multi-Market Edition
Run this file to start.

Usage:
    pip install -r requirements.txt
    python start.py
"""

import os, sys

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't encode
# the box-art / emoji / ₹ characters we print and log. Under that codepage a
# single un-encodable log line raises UnicodeEncodeError and takes the whole
# server down. Force UTF-8 (with replacement) so output can never crash us.
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Dump a full stack trace (all threads) to the console if the process hits a
# fatal native error (segfault / access violation) — the only way to see a crash
# that leaves no Python traceback.
try:
    import faulthandler
    faulthandler.enable()
except Exception:
    pass

from dotenv import load_dotenv

load_dotenv()

from app import app, initialize

print()
print("  ╔═══════════════════════════════════════════╗")
print("  ║  SignalScope v4.1 — Multi-Market Scanner  ║")
print("  ╠═══════════════════════════════════════════╣")
print("  ║  Markets: NIFTY 500 + NASDAQ 100          ║")
print("  ║  Data: Yahoo Finance (no API key needed)  ║")
print("  ╚═══════════════════════════════════════════╝")
print()

initialize()

port = int(os.getenv("PORT", 5000))
print()
print(f"  ✅ Dashboard running at: http://localhost:{port}")
print(f"     Open this URL in your browser.")
print()
print("  Press Ctrl+C to stop.")
print()

app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)