"""
SignalScope — Run this file to start.

Usage:
    python start.py
"""

import os
from dotenv import load_dotenv

# Load your .env file first
load_dotenv()

from app import app, initialize

print()
print("  ╔═══════════════════════════════════════╗")
print("  ║     SignalScope — Stock Scanner        ║")
print("  ╚═══════════════════════════════════════╝")
print()

ok = initialize()

if not ok:
    print()
    print("  ⚠  Could not connect to Angel One.")
    print("     Open http://localhost:5000 to see what's wrong.")
    print("     Fix your .env file and restart.")
    print()

port = int(os.getenv("PORT", 5000))
print()
print(f"  ✅ Dashboard running at: http://localhost:{port}")
print(f"     Open this URL in your browser.")
print()
print("  Press Ctrl+C to stop.")
print()

app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)