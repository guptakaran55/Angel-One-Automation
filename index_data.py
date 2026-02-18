"""
Index constituent lists for Indian stock markets.
Known index members are hardcoded. All other NSE equities are scanned too
and auto-classified as "OTHER NSE".
Last updated: Feb 2026.
"""

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

NIFTY_MIDCAP_100 = [
    "ADANIGREEN","ADANIPOWER","ADANIENERGY","ADANITOTAL",
    "MUTHOOTFIN","CUMMINSIND","POLYCAB","PERSISTENT","SUZLON",
    "FEDERALBNK","INDUSTOWER","INDIANB","BSE","ASHOKLEY",
    "BHEL","PBFINTECH","SWIGGY","HDFCAMC","COFORGE",
    "AUBANK","IDFCFIRSTB","OBEROIRLTY","AUROPHARMA","OFSS",
    "PIIND","ESCORTS","PAGEIND","MPHASIS","TATAELXSI",
    "KPITTECH","VOLTAS","LODHA","DELHIVERY","MRF",
    "SUNDARMFIN","LICHSGFIN","MFSL","NATIONALUM","HUDCO",
    "BANKINDIA","ALKEM","HINDCOPPER","SAIL","PETRONET",
    "DIXON","TATAPOWER","IREDA","GUJGASLTD","SJVN",
    "NMDC","BALKRISIND","SUPREMEIND","APLAPOLLO","PHOENIXLTD",
    "ASTRAL","JUBLFOOD","MANAPPURAM","DEEPAKNTR","CROMPTON",
    "SONACOMS","ACC","BIOCON","KALYANKJIL","METROBRAND",
    "BSOFT","PATANJALI","JSL","THERMAX","SYNGENE",
    "HONAUT","EXIDEIND","KEI","CGPOWER","PRESTIGE",
    "IPCALAB","CONCOR","BANDHANBNK","FORTIS","RBLBANK",
    "IDBIBANK","GMRAIRPORT","INDIAMART","ANGELONE","GODREJPROP",
    "ZEEL","LALPATHLAB","AJANTPHARM","TATACOMM","BDL",
    "COCHINSHIP","TATACHEM","ABCAPITAL","POONAWALLA","CAMS",
    "CLEAN","KAYNES","SUNTV","NAVINFLUOR","RVNL","TIINDIA",
]

NIFTY_MIDCAP_150_EXTRA = [
    "ENDURANCE","SOLARINDS","NAUKRI","CENTRALBK","NIACL",
    "EMAMILTD","GLAXO","AIAENG","IIFL","RAJESHEXPO",
    "ABFRL","BLUESTARLT","KANSAINER","SUNDRMFAST","EIDPARRY",
    "MAHABANK","LINDEINDIA","BRIGADE","GRINDWELL","SCHAEFFLER",
    "APTUS","JKCEMENT","ASTRAZEN","CYIENT","NUVOCO",
    "AAVAS","HAPPSTMNDS","RATNAMANI","FIVESTAR","GPPL",
    "SUMICHEM","TRIVENI","ZENSARTECH","POLYMED","FINCABLES",
    "MAHSEAMLESS","SAPPHIRE","WHIRLPOOL","DATAPATTNS","FINEORG",
    "ISEC","LATENTVIEW","RENUKA","ROUTE","SWANENERGY",
    "TIMKEN","PGHH",
]

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

NIFTY_MIDCAP_150 = NIFTY_MIDCAP_100 + NIFTY_MIDCAP_150_EXTRA

# All known index members combined
ALL_KNOWN = set(NIFTY_50 + NIFTY_NEXT_50 + BSE_100 + NIFTY_MIDCAP_150 + NIFTY_MIDCAP_100)

# ── Index categories (order matters for display) ──
INDEX_NAMES = ["NIFTY 50", "NIFTY 100", "NIFTY 200", "BSE 100", "MIDCAP 150", "OTHER NSE"]


def build_index_map():
    """Returns dict: symbol -> list of index names."""
    index_map = {}

    for sym in NIFTY_50:
        index_map.setdefault(sym, []).append("NIFTY 50")

    nifty_100 = set(NIFTY_50 + NIFTY_NEXT_50)
    for sym in nifty_100:
        index_map.setdefault(sym, [])
        if "NIFTY 100" not in index_map[sym]:
            index_map[sym].append("NIFTY 100")

    nifty_200 = nifty_100 | set(NIFTY_MIDCAP_100)
    for sym in nifty_200:
        index_map.setdefault(sym, [])
        if "NIFTY 200" not in index_map[sym]:
            index_map[sym].append("NIFTY 200")

    for sym in BSE_100:
        index_map.setdefault(sym, [])
        if "BSE 100" not in index_map[sym]:
            index_map[sym].append("BSE 100")

    for sym in NIFTY_MIDCAP_150:
        index_map.setdefault(sym, [])
        if "MIDCAP 150" not in index_map[sym]:
            index_map[sym].append("MIDCAP 150")

    return index_map


def get_indices_for(symbol, index_map):
    """Get index tags for a symbol. If not in any known index, tag as OTHER NSE."""
    tags = index_map.get(symbol, [])
    if not tags:
        return ["OTHER NSE"]
    return tags