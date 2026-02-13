"""
Index constituent lists for Indian stock markets.
Each stock is mapped to its index memberships.
Last updated: Feb 2026. These are revised semi-annually by NSE/BSE.
"""

# ── NIFTY 50 ──
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

# ── NIFTY NEXT 50 (these + NIFTY 50 = NIFTY 100) ──
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

# ── BSE 100 / SENSEX 100 (S&P BSE 100) ──
# Largely overlaps with NIFTY 100, with some BSE-specific additions
BSE_100 = NIFTY_50 + [
    "ABB","ABBOTINDIA","AMBUJACEM","BANKBARODA","BERGEPAINT",
    "BOSCHLTD","CHOLAFIN","COLPAL","DABUR","DIVISLAB",
    "DLF","GAIL","GODREJCP","HAVELLS","HINDPETRO",
    "ICICIPRULI","INDHOTEL","IOC","IRCTC","IRFC",
    "JIOFIN","JSWENERGY","LICI","LTIM","LUPIN",
    "MARICO","MAXHEALTH","NHPC","PFC","PIDILITIND",
    "PNB","RECLTD","SBICARD","SHREECEM","SHRIRAMFIN",
    "SIEMENS","SRF","TORNTPHARM","TVSMOTOR","VEDL",
    "YESBANK","ZOMATO","ZYDUSLIFE","BAJAJHLDNG","CANBK",
    "MOTHERSON","UNITDSPR","VBL","IDEA","UNIONBANK",
]

# ── NIFTY MIDCAP 150 (ranked 101-250 by market cap in NIFTY 500) ──
NIFTY_MIDCAP_150 = [
    "MUTHOOTFIN","CUMMINSIND","POLYCAB","PERSISTENT","SUZLON",
    "FEDERALBNK","INDUSTOWER","INDIANB","IDBIBANK","GMRAIRPORT",
    "BSE","ASHOKLEY","BHEL","PBFINTECH","SWIGGY",
    "HDFCAMC","COFORGE","AUBANK","IDFCFIRSTB","OBEROIRLTY",
    "AUROPHARMA","OFSS","PIIND","ESCORTS","PAGEIND",
    "MPHASIS","TATAELXSI","KPITTECH","SOLARINDS","VOLTAS",
    "NAUKRI","LODHA","DELHIVERY","MRF","SUNDARMFIN",
    "LICHSGFIN","CENTRALBK","MFSL","NATIONALUM","HUDCO",
    "BANKINDIA","ALKEM","HINDCOPPER","SAIL","PETRONET",
    "DIXON","TATAPOWER","IREDA","GUJGASLTD","SJVN",
    "NMDC","BALKRISIND","ENDURANCE","SUPREMEIND","APLAPOLLO",
    "PHOENIXLTD","NIACL","ASTRAL","EMAMILTD","GLAXO",
    "JUBLFOOD","MANAPPURAM","AIAENG","DEEPAKNTR","CROMPTON",
    "SONACOMS","ACC","BIOCON","KALYANKJIL","METROBRAND",
    "BSOFT","PATANJALI","JSL","THERMAX","IIFL",
    "SYNGENE","HONAUT","EXIDEIND","KEI","CGPOWER",
    "RAJESHEXPO","PRESTIGE","IPCALAB","CONCOR","BANDHANBNK",
    "PGHH","ABFRL","BLUESTARLT","FORTIS","RBLBANK",
    "KANSAINER","SUNDRMFAST","EIDPARRY","INDIAMART","ANGELONE",
    "MAHABANK","LINDEINDIA","GODREJPROP","BRIGADE","ZEEL",
    "GRINDWELL","LALPATHLAB","SCHAEFFLER","AJANTPHARM","TATACOMM",
    "APTUS","RVNL","TIINDIA","BDL","COCHINSHIP",
    "JKCEMENT","ASTRAZEN","CYIENT","TRENT","NHPC",
    "TATACHEM","ABCAPITAL","NUVOCO","AAVAS","HAPPSTMNDS",
    "CLEAN","KAYNES","POONAWALLA","RATNAMANI","CAMS",
    "FIVESTAR","GPPL","SUMICHEM","TRIVENI","ZENSARTECH",
    "POLYMED","FINCABLES","MAHSEAMLESS","SAPPHIRE","SUNTV",
    "WHIRLPOOL","DATAPATTNS","FINEORG","ISEC","LATENTVIEW",
    "NAVINFLUOR","RENUKA","ROUTE","SWANENERGY","TIMKEN",
]

# ── Build index membership map ──
def build_index_map():
    """Returns a dict: symbol -> list of index names it belongs to."""
    index_map = {}

    for sym in NIFTY_50:
        index_map.setdefault(sym, []).append("NIFTY 50")

    for sym in NIFTY_50 + NIFTY_NEXT_50:
        index_map.setdefault(sym, [])
        if "NIFTY 100" not in index_map[sym]:
            index_map[sym].append("NIFTY 100")

    for sym in BSE_100:
        index_map.setdefault(sym, [])
        if "BSE 100" not in index_map[sym]:
            index_map[sym].append("BSE 100")

    for sym in NIFTY_MIDCAP_150:
        index_map.setdefault(sym, [])
        if "MIDCAP 150" not in index_map[sym]:
            index_map[sym].append("MIDCAP 150")

    return index_map


def get_all_symbols():
    """Get the combined unique set of all symbols across all indices."""
    all_syms = set(NIFTY_50 + NIFTY_NEXT_50 + BSE_100 + NIFTY_MIDCAP_150)
    return all_syms


# Quick stats
if __name__ == "__main__":
    idx = build_index_map()
    all_s = get_all_symbols()
    print(f"NIFTY 50:     {len(NIFTY_50)} stocks")
    print(f"NIFTY 100:    {len(set(NIFTY_50 + NIFTY_NEXT_50))} stocks")
    print(f"BSE 100:      {len(set(BSE_100))} stocks")
    print(f"MIDCAP 150:   {len(NIFTY_MIDCAP_150)} stocks")
    print(f"Total unique: {len(all_s)} stocks to scan")