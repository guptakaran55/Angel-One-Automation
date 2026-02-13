"""
Angel One Credential Test — WITH TIMESTAMP DIAGNOSTICS
Checks if clock skew is causing TOTP failures.
"""

import os
import time
import struct
import socket
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# ─────────────────────────────────────────────
# 1. NTP Time Check (no extra dependencies)
# ─────────────────────────────────────────────

def get_ntp_time(ntp_server="pool.ntp.org", timeout=5):
    """
    Get the current time from an NTP server.
    Returns Unix timestamp or None on failure.
    No external packages needed — uses raw UDP.
    """
    NTP_PACKET_FORMAT = "!12I"
    NTP_DELTA = 2208988800  # 1900-01-01 to 1970-01-01 in seconds
    NTP_PORT = 123

    try:
        # Build NTP request packet
        packet = b'\x1b' + 47 * b'\0'

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.sendto(packet, (ntp_server, NTP_PORT))
        data, _ = sock.recvfrom(1024)
        sock.close()

        if data:
            unpacked = struct.unpack(NTP_PACKET_FORMAT, data[0:struct.calcsize(NTP_PACKET_FORMAT)])
            # Transmit timestamp is at index 10 (seconds) and 11 (fraction)
            ntp_time = unpacked[10] - NTP_DELTA + unpacked[11] / (2**32)
            return ntp_time
    except Exception as e:
        print(f"   ⚠️  NTP query failed ({ntp_server}): {e}")
    return None


def check_time_sync():
    """Compare local PC time against NTP servers and report drift."""
    print("\n" + "=" * 50)
    print("  🕐 TIMESTAMP DIAGNOSTICS")
    print("=" * 50)

    local_time = time.time()
    print(f"\n   Local PC time (UTC):  {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(local_time))}")
    print(f"   Local PC time (Local): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(local_time))}")
    print(f"   Unix timestamp:        {local_time:.2f}")

    # Try multiple NTP servers
    ntp_servers = ["pool.ntp.org", "time.google.com", "time.windows.com"]
    ntp_time = None

    for server in ntp_servers:
        print(f"\n   Querying NTP server: {server}...")
        ntp_time = get_ntp_time(server)
        if ntp_time:
            print(f"   NTP time (UTC):       {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ntp_time))}")
            break

    if ntp_time:
        drift = local_time - ntp_time
        abs_drift = abs(drift)
        direction = "AHEAD" if drift > 0 else "BEHIND"

        print(f"\n   ⏱️  Clock drift: {abs_drift:.2f} seconds ({direction} of real time)")

        if abs_drift < 1:
            print("   ✅ Clock is accurate — time skew is NOT the issue")
            return drift
        elif abs_drift < 30:
            print("   ⚠️  Small drift detected — probably fine for TOTP, but worth fixing")
            return drift
        elif abs_drift < 90:
            print("   ❌ SIGNIFICANT DRIFT! This is likely causing TOTP failures!")
            print(f"      TOTP codes are valid for 30-second windows.")
            print(f"      Your clock is off by {abs_drift:.1f}s — codes may be expired by the time they reach the server.")
            return drift
        else:
            print(f"   ❌ MAJOR DRIFT ({abs_drift:.0f} seconds)! This WILL cause TOTP failures!")
            return drift
    else:
        print("\n   ❌ Could not reach any NTP server. Check internet connection.")
        print("      Cannot verify if clock skew is the issue.")
        return None


# ─────────────────────────────────────────────
# 2. TOTP Multi-Window Test
# ─────────────────────────────────────────────

def test_totp_windows():
    """
    Generate TOTP codes for previous, current, and next 30-second windows.
    This shows what code would be valid at different times.
    """
    import pyotp

    print("\n" + "=" * 50)
    print("  🔑 TOTP MULTI-WINDOW TEST")
    print("=" * 50)

    totp_secret = os.getenv("ANGEL_TOTP_TOKEN", "").strip()
    if not totp_secret:
        print("\n   ❌ ANGEL_TOTP_TOKEN not found in .env")
        return

    print(f"\n   TOTP Secret: {totp_secret[:4]}...{totp_secret[-4:]} (length={len(totp_secret)})")

    try:
        totp = pyotp.TOTP(totp_secret)
    except Exception as e:
        print(f"\n   ❌ Invalid TOTP secret: {e}")
        print("      Re-copy your TOTP secret from smartapi.angelbroking.com/enable-totp")
        return

    now = time.time()
    current_window = int(now // 30)

    print(f"\n   Current time (UTC): {time.strftime('%H:%M:%S', time.gmtime(now))}")
    print(f"   Seconds into current 30s window: {int(now % 30)}s")
    print(f"\n   {'Window':<12} {'Time (UTC)':<20} {'TOTP Code':<12} {'Status'}")
    print("   " + "-" * 60)

    for offset, label in [(-2, "t-60s"), (-1, "t-30s"), (0, "CURRENT"), (1, "t+30s"), (2, "t+60s")]:
        window_time = (current_window + offset) * 30
        code = totp.generate_otp(current_window + offset)
        time_str = time.strftime('%H:%M:%S', time.gmtime(window_time))
        marker = " ← YOUR CODE" if offset == 0 else ""
        print(f"   {label:<12} {time_str:<20} {code:<12} {marker}")

    print(f"\n   If Angel One rejects code '{totp.now()}', check if one of the")
    print(f"   adjacent codes (t-30s or t+30s) matches what Angel One expects.")
    print(f"   If t-30s or t+30s would work, your clock is skewed.")

    return totp.now()


# ─────────────────────────────────────────────
# 3. Login attempt with retry on adjacent codes
# ─────────────────────────────────────────────

def test_login_with_adjacent_codes():
    """
    Try logging in with current TOTP code first.
    If it fails with 'Invalid totp', automatically retry with t-30s and t+30s codes.
    """
    import pyotp
    from SmartApi import SmartConnect

    print("\n" + "=" * 50)
    print("  🔐 LOGIN TEST (with adjacent code retry)")
    print("=" * 50)

    api_key = os.getenv("ANGEL_API_KEY", "").strip()
    client_id = os.getenv("ANGEL_CLIENT_ID", "").strip()
    password = os.getenv("ANGEL_PASSWORD", "").strip()
    totp_secret = os.getenv("ANGEL_TOTP_TOKEN", "").strip()

    if not all([api_key, client_id, password, totp_secret]):
        print("\n   ❌ Missing credentials in .env")
        return

    totp = pyotp.TOTP(totp_secret)
    current_window = int(time.time() // 30)

    # Try current code, then previous, then next
    attempts = [
        (0, "CURRENT (t=0)"),
        (-1, "PREVIOUS (t-30s)"),
        (1, "NEXT (t+30s)"),
    ]

    obj = SmartConnect(api_key=api_key)

    for offset, label in attempts:
        code = totp.generate_otp(current_window + offset)
        print(f"\n   Attempt: {label} — code: {code}")

        try:
            data = obj.generateSession(client_id, password, code)

            if data and data.get('status'):
                print(f"   ✅ LOGIN SUCCESS with {label}!")

                if offset != 0:
                    drift_estimate = offset * 30
                    direction = "BEHIND" if offset < 0 else "AHEAD"
                    print(f"\n   💡 Your PC clock is ~{abs(drift_estimate)}s {direction} of Angel One's server.")
                    print(f"      Fix: Run 'w32tm /resync /force' in admin PowerShell")
                    print(f"      Or: Settings → Time & Language → Sync now")
                else:
                    print(f"   ✅ Clock is in sync — no adjustments needed!")

                return True
            else:
                msg = data.get('message', 'Unknown error') if data else 'No response'
                code_err = data.get('errorcode', '?') if data else '?'
                print(f"   ❌ Failed: {msg} ({code_err})")

                if 'totp' not in msg.lower():
                    # Not a TOTP error — don't bother trying other codes
                    print(f"   ⛔ Non-TOTP error — stopping retry")
                    return False

        except Exception as e:
            error_str = str(e)
            if 'Invalid totp' in error_str:
                print(f"   ❌ Invalid TOTP")
            else:
                print(f"   ❌ Error: {error_str}")
                return False

    print(f"\n   ❌ ALL THREE CODES FAILED")
    print(f"   This means clock skew is NOT the problem.")
    print(f"   Your TOTP_TOKEN secret is likely wrong or expired.")
    print(f"   → Go to: smartapi.angelbroking.com/enable-totp")
    print(f"   → Re-enable TOTP and copy the NEW secret string")
    print(f"   → Update TOTP_TOKEN in your .env file")
    return False


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Angel One Credential Test + Time Diagnostics")
    print("=" * 50)

    # Step 1: Check .env values
    print("\n1. Checking .env values...")
    api_key = os.getenv("ANGEL_API_KEY", "").strip()
    client_id = os.getenv("ANGEL_CLIENT_ID", "").strip()
    password = os.getenv("ANGEL_PASSWORD", "").strip()
    totp_secret = os.getenv("ANGEL_TOTP_TOKEN", "").strip()

    for name, val in [("ANGEL_API_KEY", api_key), ("ANGEL_CLIENT_ID", client_id), ("ANGEL_PASSWORD", password), ("ANGEL_TOTP_TOKEN", totp_secret)]:
        if val:
            display = val[:4] + "..." if len(val) > 6 else "****"
            if name == "ANGEL_CLIENT_ID":
                display = val
            print(f"   {name:<12} ✅ {display}")
        else:
            print(f"   {name:<12} ❌ NOT SET")

    if not all([api_key, client_id, password, totp_secret]):
        print("\n❌ Fix missing values in .env first.")
        return

    # Step 2: Time sync check
    drift = check_time_sync()

    # Step 3: TOTP multi-window
    test_totp_windows()

    # Step 4: Login with retry
    test_login_with_adjacent_codes()

    print("\n" + "=" * 50)
    print("  DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()