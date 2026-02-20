"""
Index constituent lists for Indian stock markets.
NIFTY 500 downloaded dynamically from NSE at startup.
Sub-indices (NIFTY 50, NIFTY 100, BSE 100) hardcoded for tagging.
"""
import requests, logging, io, csv
logger = logging.getLogger(__name__)

NIFTY_50 = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BEL","BPCL",
    "BHARTIARTL","BRITANNIA","CIPLA","COALINDIA","DRREDDY",
    "EICHERMOT","ETERNAL","GRASIM","HCLTECH","HDFCBANK",
    "HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK",
    "ITC","INDUSINDBK","INFY","JSWSTEEL","KOTAKBANK",
    "LT","M&M","MARUTI","NTPC","NESTLEIND",
    "ONGC","POWERGRID","RELIANCE","SBILIFE","SBIN",
    "SUNPHARMA","TCS","TATACONSUM","TATAMOTORS","TATASTEEL",
    "TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO",
]

NIFTY_NEXT_50 = [
    "ABB","ABBOTINDIA","AMBUJACEM","BAJAJHLDNG","BANKBARODA",
    "BERGEPAINT","BOSCHLTD","CANBK","CHOLAFIN","COLPAL",
    "DABUR","DIVISLAB","DLF","GAIL","GODREJCP",
    "HAVELLS","HINDPETRO","ICICIPRULI","INDHOTEL","IOC",
    "IRCTC","IRFC","JIOFIN","JSWENERGY","LICI",
    "LTIM","LUPIN","MARICO","MAXHEALTH","NHPC",
    "PFC","PIDILITIND","PNB","RECLTD","SBICARD",
    "SHREECEM","SHRIRAMFIN","SIEMENS","SRF","TORNTPHARM",
    "TVSMOTOR","UNIONBANK","UNITDSPR","VBL","VEDL",
    "YESBANK","ZOMATO","ZYDUSLIFE","IDEA","MOTHERSON",
]

NIFTY_100 = set(NIFTY_50 + NIFTY_NEXT_50)

BSE_100_EXTRA = [
    "ADANIGREEN","ADANIPOWER","MUTHOOTFIN","CUMMINSIND","POLYCAB",
    "PERSISTENT","SUZLON","FEDERALBNK","INDUSTOWER","BSE",
    "BHEL","HDFCAMC","COFORGE","AUROPHARMA","TATAPOWER",
    "DIXON","IREDA","ACC","BIOCON","GODREJPROP",
]
BSE_100 = set(NIFTY_50 + NIFTY_NEXT_50 + BSE_100_EXTRA)

# ── Fallback if NSE CSV download fails ──
FALLBACK_EXTRA = [
    "ADANIENERGY","ADANITOTAL","INDIANB","ASHOKLEY","PBFINTECH",
    "SWIGGY","AUBANK","IDFCFIRSTB","OBEROIRLTY","OFSS","PIIND",
    "ESCORTS","PAGEIND","MPHASIS","TATAELXSI","KPITTECH","VOLTAS",
    "LODHA","DELHIVERY","MRF","SUNDARMFIN","LICHSGFIN","MFSL",
    "NATIONALUM","HUDCO","BANKINDIA","ALKEM","HINDCOPPER","SAIL",
    "PETRONET","GUJGASLTD","SJVN","NMDC","BALKRISIND","SUPREMEIND",
    "APLAPOLLO","PHOENIXLTD","ASTRAL","JUBLFOOD","MANAPPURAM",
    "DEEPAKNTR","CROMPTON","SONACOMS","KALYANKJIL","METROBRAND",
    "BSOFT","PATANJALI","JSL","THERMAX","SYNGENE","HONAUT",
    "EXIDEIND","KEI","CGPOWER","PRESTIGE","IPCALAB","CONCOR",
    "BANDHANBNK","FORTIS","RBLBANK","IDBIBANK","GMRAIRPORT",
    "INDIAMART","ANGELONE","ZEEL","LALPATHLAB","AJANTPHARM",
    "TATACOMM","BDL","COCHINSHIP","TATACHEM","ABCAPITAL",
    "POONAWALLA","CAMS","CLEAN","KAYNES","SUNTV","NAVINFLUOR",
    "RVNL","TIINDIA","ENDURANCE","SOLARINDS","NAUKRI","CENTRALBK",
    "NIACL","EMAMILTD","GLAXO","AIAENG","IIFL","RAJESHEXPO",
    "ABFRL","BLUESTARLT","KANSAINER","SUNDRMFAST","EIDPARRY",
    "MAHABANK","LINDEINDIA","BRIGADE","GRINDWELL","SCHAEFFLER",
    "APTUS","JKCEMENT","ASTRAZEN","CYIENT","NUVOCO","AAVAS",
    "HAPPSTMNDS","RATNAMANI","FIVESTAR","GPPL","SUMICHEM",
    "TRIVENI","ZENSARTECH","POLYMED","FINCABLES","TIMKEN","PGHH",
]

FALLBACK_SET = set(NIFTY_50 + NIFTY_NEXT_50 + BSE_100_EXTRA + FALLBACK_EXTRA)

NIFTY_500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

def download_nifty500():
    """Download current NIFTY 500 from NSE. Returns set of symbols or None."""
    try:
        logger.info("Downloading NIFTY 500 list from NSE...")
        resp = requests.get(NIFTY_500_URL, timeout=30)
        resp.raise_for_status()
        text = resp.content.decode('utf-8-sig', errors='replace')
        reader = csv.DictReader(io.StringIO(text))
        symbols = set()
        for row in reader:
            sym = row.get("Symbol", "").strip()
            if sym:
                symbols.add(sym)
        if len(symbols) > 400:
            logger.info(f"NIFTY 500: {len(symbols)} stocks downloaded")
            return symbols
        logger.warning(f"Only {len(symbols)} symbols — using fallback")
        return None
    except Exception as e:
        logger.warning(f"NIFTY 500 download failed: {e} — using fallback")
        return None


def get_scan_universe():
    """Get the set of symbols to scan. Tries NIFTY 500 CSV, falls back to hardcoded ~350."""
    n500 = download_nifty500()
    if n500:
        return n500
    return FALLBACK_SET


def build_index_tags():
    """Returns dict: symbol -> list of index tags (NIFTY 50, NIFTY 100, BSE 100)."""
    tags = {}
    for sym in NIFTY_50:
        tags.setdefault(sym, []).append("NIFTY 50")
    for sym in NIFTY_100:
        tags.setdefault(sym, []).append("NIFTY 100")
    for sym in BSE_100:
        tags.setdefault(sym, []).append("BSE 100")
    return tags


def get_tags_for(symbol, tag_map):
    """Get index tags for a symbol."""
    return tag_map.get(symbol, [])
