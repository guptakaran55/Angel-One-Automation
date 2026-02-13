#!/usr/bin/env python3
"""
SignalScope — Local Runner
Run on your laptop or server. Opens dashboard + sends Telegram alerts.

Usage:
  1. pip install -r backend/requirements.txt
  2. cp .env.example .env   # fill in your credentials
  3. python run_local.py
  4. Open http://localhost:5000
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    from backend.app import app, initialize

    print("=" * 50)
    print("  SignalScope — Stock Signal Scanner")
    print("=" * 50)

    ok = initialize()

    if not ok:
        print()
        print("⚠  Could not connect to Angel One.")
        print("   The dashboard will show setup instructions.")
        print("   Fix your .env credentials and restart.")
        print()

    port = int(os.getenv("PORT", 5000))
    print(f"\n🌐 Dashboard: http://localhost:{port}")
    print("   (accessible from any device on your network)")
    print("   Press Ctrl+C to stop.\n")

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()