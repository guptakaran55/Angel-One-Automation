"""
SignalScope v4.0 — Multi-Market Edition
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
print("  ║  SignalScope v4.0 — Multi-Market Scanner  ║")
print("  ╠═══════════════════════════════════════════╣")
print("  ║  Markets: NIFTY 500 + NASDAQ 100          ║")
print("  ╚═══════════════════════════════════════════╝")
print()

ok = initialize()

if not ok:
    print()
    print("  ⚠  Could not connect to Angel One.")
    print("     NASDAQ scanning will still work via Yahoo Finance.")
    print("     Fix your .env file and restart for Indian markets.")
    print()

port = int(os.getenv("PORT", 5000))
print()
print(f"  ✅ Dashboard running at: http://localhost:{port}")
print(f"     Open this URL in your browser.")
print()
print("  Press Ctrl+C to stop.")
print()

app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)