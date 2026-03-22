"""
Index constituent lists for Indian stock markets.
NIFTY 500 downloaded dynamically from NSE at startup.
Sub-indices (NIFTY 50, NIFTY 100, BSE 100) hardcoded for tagging.

NOTE: Uses urllib (not requests) — requests doesn't honor read timeouts on Windows/Python 3.12.
NOTE: Uses polling instead of Thread.join() — Python 3.12 on Windows has a race condition
      where Thread._tstate_lock doesn't release properly, causing join() to hang.
"""
import logging, io, csv, ssl, threading, time
import urllib.request
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

NIFTY_MIDCAP_150_SYMS = [
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
    "ENDURANCE","SOLARINDS","NAUKRI","CENTRALBK","NIACL",
    "EMAMILTD","GLAXO","AIAENG","IIFL","RAJESHEXPO",
    "ABFRL","BLUESTARLT","KANSAINER","SUNDRMFAST","EIDPARRY",
    "MAHABANK","LINDEINDIA","BRIGADE","GRINDWELL","SCHAEFFLER",
    "APTUS","JKCEMENT","ASTRAZEN","CYIENT","NUVOCO",
    "AAVAS","HAPPSTMNDS","RATNAMANI","FIVESTAR","GPPL",
    "SUMICHEM","TRIVENI","ZENSARTECH","POLYMED","FINCABLES",
    "TIMKEN","PGHH",
]

NIFTY_SMALLCAP_250_SYMS = [
    "MCX","LAURUSLABS","CDSL","GLAND","JBCHEPHARM","KARURVYSYA",
    "RADICO","NARAYANA","GRSE","MRPL","MSUMI","GILLETTE",
    "DEVYANI","SUNDRMHOLD","CHOLAHLDNG","ASTERDM","GODIGIT",
    "SAPPHIRE","SWANENERGY","ROUTE","LATENTVIEW","ISEC",
    "DATAPATTNS","FINEORG","WHIRLPOOL","MAHSEAMLESS","RENUKA",
    "AFFLE","TTML","TRITURBINE","EDELWEISS","NESCO",
    "STARCEMENT","CENTURYTEX","TEAMLEASE","MAPMYINDIA",
    "MAHLIFE","NETWORK18","PPLPHARMA","WESTLIFE","OLECTRA",
    "JTLIND","BIKAJI","SAPPHIRE","TARSONS","KRSNAA",
    "GRAVITA","TANLA","ECLERX","CRAFTSMAN","SHYAMMETL",
    "VGUARD","AMIORG","JYOTHYLAB","STARHEALTH","MOTILALOFS",
    "TTKPRESTIG","UTIAMC","NUVAMA","PNBHOUSING","INDIACEM",
    "ZFCVINDIA","ELGIEQUIP","SONATSOFTW","GOCOLORS","GODFRYPHLP",
    "CHALET","KIOCL","CERA","ISGEC","TEGA","SUVENPHAR",
    "GPIL","SPARC","SBFC","PRINCEPIPE","HBLPOWER","TV18BRDCST",
    "RALLIS","FLUOROCHEM","REDINGTON","BLUEDART","POWERINDIA",
    "CARBORUNIV","CENTURYPLY","PRSMJOHNSN","FINPIPE","MEDPLUS",
    "ANURAS","SENCO","GATEWAY","RAINBOW","AARTIIND",
    "TCIEXP","MAHINDCIE","MASTEK","ASAHIINDIA","LAXMIMACH",
    "VSTIND","BALAMINES","SUPRAJIT","KPIL","QUESS",
    "AETHER","GPPL","JKLAKSHMI","NIITMTS","RAYMOND",
    "WELCORP","KIMS","RATEGAIN","SAREGAMA","MAHLOG",
    "GLENMARK","IRCON","NATCOPHARM","HINDPETRO","ZYDUSWELL",
    "JSWINFRA","SHRIRAMCIT","TRIDENT","INOXWIND","NCC",
    "DCMSHRIRAM","RKFORGE","CHEMPLASTS","GMMPFAUDLR","BLUEJET",
    "BBTC","JAMNAAUTO","AARTIDRUGS","TDPOWERSYS","GHCL",
    "SPLPETRO","THERMAX","KFINTECH","ABSLAMC","SANOFI",
    "NIACL","CRAFTSMAN","HINDWAREAP","SAKSOFT","RITES",
    "PEL","LXCHEM","NEWGEN","ACE","MASFIN",
    "ENGINERSIN","SAGCEM","IFBIND","DOMS","AVALON",
    "SBCL","ADFFOODS","VAIBHAVGBL","SAFARI","INDIGOPNTS",
    "SHOPERSTOP","GREENPANEL","PURMO","UCOBANK","ZOMATO",
    "CUB","MAZDOCK","BANARISUG","TATAINVEST","RAMCOCEM",
    "JKPAPER","SFL","CCL","LUXIND","GARFIBRES",
    "MIDHANI","DATAMATICS","VIPIND","GILLETTE","TVSSRICHAK",
    "CASTROLIND","GULFOILLUB","NESCO","ALKYLAMINE","ELECON",
    "PRAJIND","BAJAJELEC","PNCINFRA","IRFC","SHILPAMED",
    "WOCKPHARMA","GESHIP","ORIENTELEC","GREENLAM","JKTYRE",
    "DALBHARAT","SKFINDIA","SANOFI","GMDCLTD","CAPACITE",
    "SYMPHONY","SUNTECK","WESTLIFE","SYRMA","INTELLECT",
]

FALLBACK_SET = set(NIFTY_50 + NIFTY_NEXT_50 + BSE_100_EXTRA +
                    NIFTY_MIDCAP_150_SYMS + NIFTY_SMALLCAP_250_SYMS)

NIFTY_500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
]

def download_nifty500():
    """Download current NIFTY 500 from NSE using urllib.
    Uses polling instead of Thread.join() to avoid Windows/Python 3.12 tstate_lock bug."""
    result = [None]
    done = [False]

    def _try_download():
        for url in NIFTY_500_URLS:
            try:
                logger.info(f"Trying NIFTY 500 from: {url}")
                ctx = ssl.create_default_context()
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0",
                    "Accept": "text/csv,text/plain,*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                })
                with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                    raw = resp.read()
                text = raw.decode('utf-8-sig', errors='replace')
                reader = csv.DictReader(io.StringIO(text))
                symbols = set()
                for row in reader:
                    sym = (row.get("Symbol") or row.get("symbol") or row.get("SYMBOL") or "").strip()
                    if sym:
                        symbols.add(sym)
                if len(symbols) > 400:
                    logger.info(f"NIFTY 500: {len(symbols)} stocks from {url}")
                    result[0] = symbols
                    done[0] = True
                    return
                logger.warning(f"Only {len(symbols)} symbols from {url}")
            except Exception as e:
                logger.warning(f"Failed {url}: {e}")
        done[0] = True

    threading.Thread(target=_try_download, daemon=True).start()

    # Poll instead of join — avoids tstate_lock bug on Windows/Python 3.12
    for _ in range(40):  # 40 x 0.5s = 20s max
        if done[0]:
            break
        time.sleep(0.5)

    if not done[0]:
        logger.warning("NIFTY 500 download timed out after 20s — using fallback list")
        return None
    return result[0]


def get_scan_universe():
    n500 = download_nifty500()
    if n500:
        return n500
    logger.warning("Using expanded fallback list (~350 stocks)")
    return FALLBACK_SET


def build_index_tags():
    tags = {}
    for sym in NIFTY_50:
        tags.setdefault(sym, []).append("NIFTY 50")
    for sym in NIFTY_100:
        tags.setdefault(sym, []).append("NIFTY 100")
    for sym in BSE_100:
        tags.setdefault(sym, []).append("BSE 100")
    return tags


def get_tags_for(symbol, tag_map):
    return tag_map.get(symbol, [])