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

        ticker = yf.Ticker(symbol)
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
    Fetch index chart data for display.
    
    Args:
        index_symbol: Yahoo Finance symbol (^NSEI for Nifty, ^IXIC for NASDAQ, etc.)
        resolution: "1d", "1wk", or "1mo"
        period: "1mo", "3mo", "6mo", "1y", "2y", "5y"
    
    Returns dict with dates and values for charting.
    """
    try:
        import yfinance as yf

        valid_resolutions = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
        interval = valid_resolutions.get(resolution, "1d")

        # Adjust period based on resolution to get sensible data
        if interval == "1d" and period not in ["1mo", "3mo", "6mo", "1y"]:
            period = "1y"
        elif interval == "1wk" and period not in ["3mo", "6mo", "1y", "2y"]:
            period = "1y"
        elif interval == "1mo" and period not in ["1y", "2y", "5y"]:
            period = "5y"

        ticker = yf.Ticker(index_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df is None or len(df) < 2:
            return None

        df = df.dropna(subset=["Close"])

        # Calculate stats
        latest = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else latest
        first = float(df["Close"].iloc[0])
        high_val = float(df["High"].max())
        low_val = float(df["Low"].min())
        day_change = latest - prev
        day_change_pct = (day_change / prev * 100) if prev > 0 else 0
        period_change = latest - first
        period_change_pct = (period_change / first * 100) if first > 0 else 0

        # Build chart data
        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        closes = [round(float(v), 2) for v in df["Close"].values]
        highs = [round(float(v), 2) for v in df["High"].values]
        lows = [round(float(v), 2) for v in df["Low"].values]
        volumes = [int(v) for v in df["Volume"].values]

        # Index metadata
        INDEX_META = {
            "^NSEI": {"name": "NIFTY 50", "currency": "₹", "exchange": "NSE"},
            "^NSEBANK": {"name": "BANK NIFTY", "currency": "₹", "exchange": "NSE"},
            "^BSESN": {"name": "SENSEX", "currency": "₹", "exchange": "BSE"},
            "^IXIC": {"name": "NASDAQ Composite", "currency": "$", "exchange": "NASDAQ"},
            "^NDX": {"name": "NASDAQ 100", "currency": "$", "exchange": "NASDAQ"},
            "^GSPC": {"name": "S&P 500", "currency": "$", "exchange": "NYSE"},
            "^DJI": {"name": "Dow Jones", "currency": "$", "exchange": "NYSE"},
            # Indian Sector Indices (Yahoo Finance tickers)
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
        logger.error(f"Index chart error for {index_symbol}: {e}")
        return None