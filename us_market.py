"""
US Market Data Module — yfinance integration for SignalScope.
Provides NASDAQ 100 constituent list, candle data fetching, and index chart data.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

US_RATE_LIMITED = "US_RATE_LIMITED"

# ═══════════════════════════════════════════════════════════════
# NASDAQ 100 CONSTITUENTS (updated March 2026)
# ═══════════════════════════════════════════════════════════════

NASDAQ_100 = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COIN", "COST", "CPRT", "CRWD",
    "CSCO", "CSGP", "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM", "EA", "EXC",
    "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX",
    "ILMN", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU",
    "MAR", "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MU",
    "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD",
    "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS",
    "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY",
    "XEL", "ZS",
]


def get_nasdaq100_symbols():
    """Return list of NASDAQ 100 symbols."""
    return NASDAQ_100.copy()


def fetch_us_candle_data(symbol, period="2y"):
    """
    Fetch daily OHLCV data from Yahoo Finance.
    Returns DataFrame with same format as Angel One data, None, or US_RATE_LIMITED.
    """
    try:
        import yfinance as yf
        import requests as _req

        session = _req.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0",
        })

        ticker = yf.Ticker(symbol, session=session)
        df = ticker.history(period=period, interval="1d", auto_adjust=True)

        if df is None or len(df) < 10:
            return None

        # Rename columns to match Angel One format
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        })

        # Keep only the columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Drop any rows with NaN
        df = df.dropna()

        if len(df) < 50:
            return None

        return df

    except Exception as e:
        err_str = str(e).lower()
        if "rate" in err_str or "too many" in err_str or "429" in err_str:
            logger.warning(f"US rate limited on {symbol}: {e}")
            return US_RATE_LIMITED
        logger.debug(f"US candle error {symbol}: {e}")
        return None


def fetch_index_chart_data(index_symbol="^NSEI", resolution="1d", period="1y"):
    """
    Fetch index chart data using Yahoo Finance v8 chart API directly.
    Bypasses yfinance library to avoid its aggressive rate-limit detection
    (the _fetch_ticker_tz call that triggers YFRateLimitError).
    """
    import requests as _req

    valid_resolutions = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
    interval = valid_resolutions.get(resolution, "1d")

    # Adjust period based on resolution
    if interval == "1d" and period not in ["1mo", "3mo", "6mo", "1y"]:
        period = "1y"
    elif interval == "1wk" and period not in ["3mo", "6mo", "1y", "2y"]:
        period = "1y"
    elif interval == "1mo" and period not in ["1y", "2y", "5y"]:
        period = "5y"

    # Yahoo v8 chart API — no auth, no tz lookup, much less likely to rate-limit
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{index_symbol}"
    params = {
        "range": period,
        "interval": interval,
        "includePrePost": "false",
        "events": "",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }

    try:
        logger.info(f"Yahoo chart API: fetching {index_symbol} period={period} interval={interval}")
        resp = _req.get(url, params=params, headers=headers, timeout=15)

        if resp.status_code == 429:
            logger.warning(f"Yahoo chart API rate limited for {index_symbol}")
            return None
        if resp.status_code != 200:
            logger.warning(f"Yahoo chart API returned {resp.status_code} for {index_symbol}")
            return None

        data = resp.json()
        chart = data.get("chart", {}).get("result", [])
        if not chart:
            logger.warning(f"Yahoo chart API: no result for {index_symbol}")
            return None

        result = chart[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        closes_raw = quote.get("close", [])
        highs_raw = quote.get("high", [])
        lows_raw = quote.get("low", [])
        volumes_raw = quote.get("volume", [])

        if len(timestamps) < 2 or len(closes_raw) < 2:
            logger.warning(f"Yahoo chart API: insufficient data for {index_symbol} ({len(timestamps)} points)")
            return None

        # Build clean arrays, skipping None values
        from datetime import datetime
        dates = []
        closes = []
        highs = []
        lows = []
        volumes = []
        for i, ts in enumerate(timestamps):
            c = closes_raw[i] if i < len(closes_raw) else None
            if c is None:
                continue
            dates.append(datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"))
            closes.append(round(float(c), 2))
            highs.append(round(float(highs_raw[i] or c), 2))
            lows.append(round(float(lows_raw[i] or c), 2))
            volumes.append(int(volumes_raw[i] or 0))

        if len(closes) < 2:
            return None

        latest = closes[-1]
        prev = closes[-2]
        first = closes[0]
        high_val = max(highs)
        low_val = min(lows)
        day_change = latest - prev
        day_change_pct = (day_change / prev * 100) if prev > 0 else 0
        period_change = latest - first
        period_change_pct = (period_change / first * 100) if first > 0 else 0

        logger.info(f"Yahoo chart API OK: {index_symbol} — {len(dates)} points")

        # Index metadata
        INDEX_META = {
            "^NSEI": {"name": "NIFTY 50", "currency": "₹", "exchange": "NSE"},
            "^NSEBANK": {"name": "BANK NIFTY", "currency": "₹", "exchange": "NSE"},
            "^BSESN": {"name": "SENSEX", "currency": "₹", "exchange": "BSE"},
            "^IXIC": {"name": "NASDAQ Composite", "currency": "$", "exchange": "NASDAQ"},
            "^NDX": {"name": "NASDAQ 100", "currency": "$", "exchange": "NASDAQ"},
            "^GSPC": {"name": "S&P 500", "currency": "$", "exchange": "NYSE"},
            "^DJI": {"name": "Dow Jones", "currency": "$", "exchange": "NYSE"},
            "^CNXBANK": {"name": "NIFTY Bank", "currency": "₹", "exchange": "NSE", "sector": "Banking"},
            "^CNXIT": {"name": "NIFTY IT", "currency": "₹", "exchange": "NSE", "sector": "IT / Tech"},
            "^CNXPHARMA": {"name": "NIFTY Pharma", "currency": "₹", "exchange": "NSE", "sector": "Pharma"},
            "^CNXFMCG": {"name": "NIFTY FMCG", "currency": "₹", "exchange": "NSE", "sector": "FMCG"},
            "^CNXAUTO": {"name": "NIFTY Auto", "currency": "₹", "exchange": "NSE", "sector": "Auto"},
            "^CNXENERGY": {"name": "NIFTY Energy", "currency": "₹", "exchange": "NSE", "sector": "Energy"},
            "^CNXFIN": {"name": "NIFTY Fin Service", "currency": "₹", "exchange": "NSE", "sector": "Financial Services"},
            "^CNXMETAL": {"name": "NIFTY Metal", "currency": "₹", "exchange": "NSE", "sector": "Metals"},
            "^CNXREALTY": {"name": "NIFTY Realty", "currency": "₹", "exchange": "NSE", "sector": "Realty"},
            "^CNXMEDIA": {"name": "NIFTY Media", "currency": "₹", "exchange": "NSE", "sector": "Media"},
        }
        meta = INDEX_META.get(index_symbol, {"name": index_symbol, "currency": "", "exchange": ""})

        return {
            "symbol": index_symbol,
            "name": meta["name"],
            "currency": meta["currency"],
            "exchange": meta["exchange"],
            "resolution": resolution,
            "period": period,
            "latest": round(latest, 2),
            "day_change": round(day_change, 2),
            "day_change_pct": round(day_change_pct, 2),
            "period_change": round(period_change, 2),
            "period_change_pct": round(period_change_pct, 2),
            "high": round(high_val, 2),
            "low": round(low_val, 2),
            "data_points": len(dates),
            "dates": dates,
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "volumes": volumes,
        }

    except Exception as e:
        logger.error(f"Index chart error for {index_symbol}: {type(e).__name__}: {e}", exc_info=True)
        return None
