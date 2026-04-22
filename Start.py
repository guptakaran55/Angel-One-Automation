"""
SignalScope v4.1 — Multi-Market Edition
Run this file to start.

Usage:
    pip install -r requirements.txt
    python start.py
"""

import os
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