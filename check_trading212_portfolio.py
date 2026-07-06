"""
Console check for the Trading 212 Public API.

Usage:
    python check_trading212_portfolio.py --env demo
    python check_trading212_portfolio.py --env live

Credentials are read from:
    T212_API_KEY / TRADING212_API_KEY
    T212_API_SECRET / TRADING212_API_SECRET

If they are not present, the script prompts for them without saving anything.
The prompt is visible by default because hidden Windows prompts can be awkward
inside the VS Code terminal.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URLS = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def load_dotenv_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ask_value(label: str, hidden: bool) -> str:
    if hidden:
        return getpass.getpass(f"{label}: ").strip()
    return input(f"{label}: ").strip()


def get_credentials(hidden_prompt: bool = False) -> tuple[str, str]:
    api_key = first_env("T212_API_KEY", "TRADING212_API_KEY")
    api_secret = first_env("T212_API_SECRET", "TRADING212_API_SECRET")

    if not api_key:
        api_key = ask_value("Trading 212 API key", hidden_prompt)
    if not api_secret:
        api_secret = ask_value("Trading 212 API secret", hidden_prompt)

    if not api_key or not api_secret:
        raise ValueError("Both API key and API secret are required.")

    return api_key, api_secret


def request_json(base_url: str, path: str, auth_header: str, environment: str) -> Any:
    request = Request(
        f"{base_url}{path}",
        headers={
            "Accept": "application/json",
            "Authorization": auth_header,
            "User-Agent": "trading212-console-check/1.0",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=20) as response:
            rate_remaining = response.headers.get("x-ratelimit-remaining")
            if rate_remaining is not None:
                print(f"Rate limit remaining for {path}: {rate_remaining}")

            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 401:
            other_env = "live" if environment == "demo" else "demo"
            raise RuntimeError(
                "Unauthorized. Check your API key and secret. "
                f"If these keys were generated for {other_env.upper()}, run again with --env {other_env}."
            ) from exc
        if exc.code == 403:
            raise RuntimeError("Forbidden. Check API permissions, account type, or IP restrictions.") from exc
        if exc.code == 429:
            raise RuntimeError("Rate limited. Wait a few seconds and try again.") from exc

        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def money(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def print_mapping(title: str, data: Any) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not isinstance(data, dict):
        print(data)
        return

    for key in sorted(data):
        print(f"{key}: {data[key]}")


def index_instruments(instruments: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(instruments, list):
        return {}

    indexed: dict[str, dict[str, Any]] = {}
    for item in instruments:
        if isinstance(item, dict) and item.get("ticker"):
            indexed[str(item["ticker"])] = item
    return indexed


def print_portfolio(positions: Any, instruments_by_ticker: dict[str, dict[str, Any]] | None = None) -> None:
    instruments_by_ticker = instruments_by_ticker or {}

    print("\nOpen Positions")
    print("--------------")

    if not positions:
        print("No open positions returned.")
        return
    if not isinstance(positions, list):
        print(positions)
        return

    headers = [
        "ticker",
        "name",
        "isin",
        "currency",
        "quantity",
        "avg_price",
        "current_price",
        "p_l",
    ]
    rows: list[list[str]] = []

    for item in positions:
        if not isinstance(item, dict):
            rows.append([str(item), "", "", "", "", ""])
            continue

        ticker = str(item.get("ticker", "-"))
        instrument = instruments_by_ticker.get(ticker, {})
        rows.append(
            [
                ticker,
                str(instrument.get("shortName") or instrument.get("name") or item.get("frontend") or "-"),
                str(instrument.get("isin", "-")),
                str(instrument.get("currencyCode", "-")),
                money(item.get("quantity")),
                money(item.get("averagePrice")),
                money(item.get("currentPrice")),
                money(item.get("ppl")),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    total_p_l = sum(
        item.get("ppl", 0)
        for item in positions
        if isinstance(item, dict) and isinstance(item.get("ppl"), (int, float))
    )
    print(f"\nPositions: {len(positions)}")
    print(f"Total unrealized P/L from returned positions: {money(total_p_l)}")


def main() -> int:
    load_dotenv_file()

    parser = argparse.ArgumentParser(description="Check Trading 212 API connection and print portfolio.")
    parser.add_argument(
        "--env",
        choices=BASE_URLS,
        default=os.getenv("T212_ENV", "demo").lower(),
        help="Trading 212 environment to use. Default: demo, or T212_ENV if set.",
    )
    parser.add_argument(
        "--skip-cash",
        action="store_true",
        help="Only fetch portfolio positions.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not fetch instrument metadata for stock names, ISINs, and currencies.",
    )
    parser.add_argument(
        "--hidden-prompt",
        action="store_true",
        help="Hide credential input while typing. Default is visible input for VS Code compatibility.",
    )
    args = parser.parse_args()

    api_key, api_secret = get_credentials(hidden_prompt=args.hidden_prompt)
    base_url = BASE_URLS[args.env]
    credentials = f"{api_key}:{api_secret}".encode("utf-8")
    auth_header = f"Basic {base64.b64encode(credentials).decode('ascii')}"

    print(f"Checking Trading 212 {args.env.upper()} API at {base_url}")

    try:
        if not args.skip_cash:
            cash = request_json(base_url, "/equity/account/cash", auth_header, args.env)
            print_mapping("Account Cash", cash)

        instruments_by_ticker = {}
        if not args.skip_metadata:
            instruments = request_json(base_url, "/equity/metadata/instruments", auth_header, args.env)
            instruments_by_ticker = index_instruments(instruments)

        portfolio = request_json(base_url, "/equity/portfolio", auth_header, args.env)
        print_portfolio(portfolio, instruments_by_ticker)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    print("\nConnection check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
