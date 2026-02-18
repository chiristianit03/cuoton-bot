import os
import math
import time
import json
import difflib
import requests
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =========================
# ENV
# =========================
ODDS_API_KEY = os.environ["ODDS_API_KEY"]
TG_BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]
GITHUB_SHA = os.environ.get("GITHUB_SHA", "")[:7]

# API-Football (autodetección)
# - Si defines APIFOOTBALL_KEY => usa API-Sports directo (https://v3.football.api-sports.io)
# - Si defines RAPIDAPI_KEY y RAPIDAPI_HOST => usa RapidAPI
APIFOOTBALL_KEY = os.environ.get("APIFOOTBALL_KEY", "").strip()
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "").strip()
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com").strip()

# Odds API
ODDS_FORMAT = "decimal"
REGIONS_PRIMARY = "uk"
REGIONS_FALLBACK = ["uk", "eu"]  # si quieres añadir "us", te puede gastar más

SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
]

# API-Football league ids (estándar)
LEAGUE_MAP = {
    "soccer_epl": 39,              # Premier League
    "soccer_spain_la_liga": 140,   # La Liga
    "soccer_germany_bundesliga": 78,  # Bundesliga
}

# =========================
# OBJETIVOS
# =========================
TARGET_900 = 900.0
TARGET_300 = 300.0

LEGS_900 = (4, 6)
LEGS_300 = (4, 6)

ODDS_TOL_900 = 0.18   # ~738 a 1062
ODDS_TOL_300 = 0.22

BEAM_WIDTH = 900

# Cuotas por pata (global)
ODDS_MIN = 1.25
ODDS_MAX = 25.0

# Para cuotón (alto pero no loco)
CUOTON_ODDS_MIN = 3.0
CUOTON_ODDS_MAX = 7.5

# Min prob por pata (final)
MIN_P_LEG_CUOTON = 0.21

# Eventos enriquecidos (OJO créditos)
MAX_EVENTS_ENRICH = 10
EVENT_ODDS_SLEEP_SEC = 0.18

# Mezcla modelo vs mercado
MODEL_WEIGHT = 0.55  # p_final = w*p_model + (1-w)*p_market

# Beam tuning
LAMBDA_ODDS = 0.35
TARGET_TIGHTNESS = 1.35

# =========================
# MARKETS
# =========================
# Pedimos corners/cards primero (para maximizar chance)
CORNERS_CARDS_MARKETS = [
    "alternate_totals_corners",
    "alternate_spreads_corners",
    "alternate_totals_cards",
    "alternate_spreads_cards",
]

# Core markets (los que ya te devolvía)
CORE_MARKETS = [
    "alternate_totals",
    "alternate_spreads",
    "btts",
    "draw_no_bet",
    "double_chance",
    "h2h_3_way",
    "team_totals",
    "alternate_team_totals",
]

ALL_EVENT_MARKETS = CORNERS_CARDS_MARKETS + CORE_MARKETS


# =========================
# MODELS / DATA
# =========================
@dataclass(frozen=True)
class Pick:
    match_id: str
    sport_key: str
    market: str
    outcome: str
    point: Optional[float]
    odds_best: float
    odds_median: float
    p_market: float
    p_model: Optional[float]
    p_final: float
    value_ratio: float
    label: str


# =========================
# TELEGRAM
# =========================
def tg_send(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg}, timeout=25)
    except Exception:
        pass


# =========================
# HELPERS
# =========================
def median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    m = n // 2
    return xs[m] if n % 2 == 1 else 0.5 * (xs[m - 1] + xs[m])

def safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))

def odds_product(acca: List[Pick]) -> float:
    prod = 1.0
    for p in acca:
        prod *= p.odds_best
    return prod

def normalize_team(s: str) -> str:
    s = s.lower()
    for ch in [".", ",", "-", "_", "(", ")", "'", '"', "’"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    # pequeñas normalizaciones útiles
    s = s.replace("fc", "").replace("cf", "").replace("sc", "").replace("sv", "")
    return " ".join(s.split())

def fuzzy_match(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_team(a), normalize_team(b)).ratio()


# =========================
# MARKET PREFERENCES / FILTERS
# =========================
def market_weight(mk: str) -> float:
    return {
        "alternate_totals_corners": 0.30,
        "alternate_spreads_corners": 0.26,
        "alternate_totals_cards": 0.26,
        "alternate_spreads_cards": 0.22,
        "alternate_totals": 0.20,
        "alternate_spreads": 0.18,
        "btts": 0.14,
        "team_totals": 0.12,
        "alternate_team_totals": 0.12,
        "draw_no_bet": 0.08,
        "double_chance": 0.06,
        "h2h_3_way": -0.08,
    }.get(mk, 0.0)

def is_draw_like(mk: str, outcome: str) -> bool:
    return mk == "h2h_3_way" and outcome.lower() == "draw"

def is_extreme_line(mk: str, point: Optional[float]) -> bool:
    if point is None:
        return False

    # Goals totals (alternate_totals)
    if mk == "alternate_totals":
        if point < 1.5 or point > 5.5:
            return True

    # Team totals (goles por equipo)
    if mk in ("team_totals", "alternate_team_totals"):
        if point < 0.5 or point > 3.5:
            return True

    # Corners totals
    if mk == "alternate_totals_corners":
        if point < 7.5 or point > 14.5:
            return True

    # Cards totals
    if mk == "alternate_totals_cards":
        if point < 2.5 or point > 9.5:
            return True

    # Asian handicap (spreads)
    if mk == "alternate_spreads":
        if point < -2.5 or point > 2.5:
            return True

    # Corner/Card handicaps
    if mk in ("alternate_spreads_corners", "alternate_spreads_cards"):
        if point < -4.5 or point > 4.5:
            return True

    return False


# =========================
# ODDS API CALLS
# =========================
def fetch_odds_list(sport_key: str, region: str = REGIONS_PRIMARY) -> list:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": "h2h",
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    used = r.headers.get("x-requests-used")
    rem = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        tg_send(f"❌ /odds ERROR {sport_key} HTTP {r.status_code} used={used} rem={rem}\n{r.text[:280]}")
        return []
    data = r.json()
    bm0 = len(data[0].get("bookmakers", [])) if data else 0
    tg_send(f"✅ /odds {sport_key}: events={len(data)} first_bookmakers={bm0} used={used} rem={rem} sha={GITHUB_SHA}")
    return data

def fetch_event_odds(sport_key: str, event_id: str, markets: List[str], region: str) -> Optional[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    used = r.headers.get("x-requests-used")
    rem = r.headers.get("x-requests-remaining")
    last = r.headers.get("x-requests-last")
    if r.status_code != 200:
        tg_send(f"❌ event_odds ERROR {sport_key} {event_id} reg={region} HTTP {r.status_code} last={last} used={used} rem={rem}\n{r.text[:280]}")
        return None

    data = r.json()
    mk = set()
    for bm in data.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m.get("key"):
                mk.add(m.get("key"))
    tg_send(f"🧩 event_odds OK {sport_key} reg={region} returned={sorted(mk)[:18]} last={last} used={used} rem={rem}")
    return data

def extract_market_keys(event_data: dict) -> set:
    mk = set()
    for bm in event_data.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m.get("key"):
                mk.add(m.get("key"))
    return mk

def fetch_event_odds_best_effort(sport_key: str, event_id: str) -> Tuple[Optional[dict], set]:
    """
    1) intenta corners/cards primero (por región uk->eu)
    2) luego core markets
    Devuelve (data_final, markets_seen)
    """
    markets_seen = set()
    best_data = None

    # A) corners/cards-first
    corners_ok = False
    for reg in REGIONS_FALLBACK:
        d = fetch_event_odds(sport_key, event_id, CORNERS_CARDS_MARKETS, reg)
        time.sleep(EVENT_ODDS_SLEEP_SEC)
        if not d:
            continue
        mk = extract_market_keys(d)
        markets_seen |= mk
        if any(k in mk for k in CORNERS_CARDS_MARKETS):
            corners_ok = True
            best_data = d
            break

    # B) core markets (si corners no llegaron, igual traemos core)
    for reg in REGIONS_FALLBACK:
        d2 = fetch_event_odds(sport_key, event_id, CORE_MARKETS, reg)
        time.sleep(EVENT_ODDS_SLEEP_SEC)
        if not d2:
            continue
        mk2 = extract_market_keys(d2)
        markets_seen |= mk2

        # “merge” simple: si ya teníamos corners_data, añadimos markets core
        if best_data is None:
            best_data = d2
        else:
            # merge bookmakers markets
            # (simple: concatenamos bookmakers; extractor usa mediana/max, tolera duplicados)
            best_data["bookmakers"] = (best_data.get("bookmakers", []) or []) + (d2.get("bookmakers", []) or [])

        break

    return best_data, markets_seen

def fav_strength_from_h2h(event: dict) -> float:
    best_fav_odds = 999.0
    for bm in event.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes", []):
                price = o.get("price")
                if isinstance(price, (int, float)):
                    best_fav_odds = min(best_fav_odds, float(price))
    if best_fav_odds >= 999.0:
        return 0.0
    return 1.0 / best_fav_odds


# =========================
# API-FOOTBALL CLIENT (autodetect)
# =========================
class APIFootball:
    def __init__(self):
        self.mode = None
        self.base = None
        self.headers = {}

        if APIFOOTBALL_KEY:
            self.mode = "apisports"
            self.base = "https://v3.football.api-sports.io"
            self.headers = {"x-apisports-key": APIFOOTBALL_KEY}
        elif RAPIDAPI_KEY:
            self.mode = "rapidapi"
            self.base = "https://api-football-v1.p.rapidapi.com/v3"
            self.headers = {
                "X-RapidAPI-Key": RAPIDAPI_KEY,
                "X-RapidAPI-Host": RAPIDAPI_HOST,
            }

        # cache en memoria (por corrida)
        self.cache: Dict[str, dict] = {}

    def enabled(self) -> bool:
        return self.mode is not None

    def _get(self, path: str, params: dict) -> Optional[dict]:
        if not self.enabled():
            return None
        key = f"{path}?{json.dumps(params, sort_keys=True)}"
        if key in self.cache:
            return self.cache[key]

        url = f"{self.base}{path}"
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=25)
            if r.status_code != 200:
                tg_send(f"⚠️ API-Football HTTP {r.status_code} {path} {str(r.text)[:200]}")
                return None
            data = r.json()
            self.cache[key] = data
            return data
        except Exception as e:
            tg_send(f"⚠️ API-Football error {path}: {e}")
            return None

    def status(self) -> Optional[dict]:
        return self._get("/status", {})

    def fixtures_by_date_league(self, date_yyyy_mm_dd: str, league_id: int, season: int) -> List[dict]:
        data = self._get("/fixtures", {"date": date_yyyy_mm_dd, "league": league_id, "season": season})
        if not data or "response" not in data:
            return []
        return data["response"]

    def team_statistics(self, league_id: int, season: int, team_id: int) -> Optional[dict]:
        data = self._get("/teams/statistics", {"league": league_id, "season": season, "team": team_id})
        if not data or "response" not in data:
            return None
        return data["response"]  # dict

def season_for_date(dt: datetime.datetime) -> int:
    # Europa: temporada suele ser año de inicio (2025 para 2025/26)
    return dt.year - 1 if dt.month <= 6 else dt.year


# =========================
# MODELOS (MVP)
# =========================
def poisson_p_over(lmb: float, line: float) -> float:
    """
    P(X > line) para X~Poisson(lmb) y line tipo 2.5, 3.5...
    Over 2.5 = P(X>=3). line puede ser .5 -> k = floor(line)+1
    """
    k = int(math.floor(line) + 1)
    # P(X >= k) = 1 - sum_{i=0..k-1} e^-λ λ^i / i!
    s = 0.0
    for i in range(k):
        s += math.exp(-lmb) * (lmb ** i) / math.factorial(i)
    return max(0.0, min(1.0, 1.0 - s))

def poisson_p_under(lmb: float, line: float) -> float:
    """
    Under 2.5 = P(X<=2)
    """
    k = int(math.floor(line))
    s = 0.0
    for i in range(k + 1):
        s += math.exp(-lmb) * (lmb ** i) / math.factorial(i)
    return max(0.0, min(1.0, s))

def btts_prob(lh: float, la: float) -> float:
    # P(home>=1 and away>=1) = (1-e^-lh)*(1-e^-la)
    return (1.0 - math.exp(-lh)) * (1.0 - math.exp(-la))

def get_goal_lambdas(team_stats_home: dict, team_stats_away: dict) -> Optional[Tuple[float, float]]:
    """
    Intento robusto de leer goles a favor/en contra por partido.
    API-Football teams/statistics trae varias estructuras; aquí intentamos algunas rutas típicas.
    Si no encontramos, devolvemos None.
    """
    try:
        # promedio total
        gf_home = team_stats_home["goals"]["for"]["average"]["total"]
        ga_home = team_stats_home["goals"]["against"]["average"]["total"]
        gf_away = team_stats_away["goals"]["for"]["average"]["total"]
        ga_away = team_stats_away["goals"]["against"]["average"]["total"]
        # strings tipo "1.42"
        gf_home = float(gf_home)
        ga_home = float(ga_home)
        gf_away = float(gf_away)
        ga_away = float(ga_away)

        # lambdas mezcla simple
        lam_home = (gf_home + ga_away) / 2.0
        lam_away = (gf_away + ga_home) / 2.0

        # sanity
        if lam_home <= 0 or lam_away <= 0:
            return None
        return lam_home, lam_away
    except Exception:
        return None

def estimate_total_corners_lambda(team_stats_home: dict, team_stats_away: dict) -> Optional[float]:
    """
    Corner stats no siempre están. Intentamos algunas rutas comunes.
    Si no existe, None.
    """
    # No garantizado en todos los planes / ligas
    candidates = [
        ("corners", "for", "average", "total"),
        ("corners", "against", "average", "total"),
    ]
    try:
        # si existiera corners for/against avg, usamos mezcla
        cf_home = float(team_stats_home["corners"]["for"]["average"]["total"])
        ca_home = float(team_stats_home["corners"]["against"]["average"]["total"])
        cf_away = float(team_stats_away["corners"]["for"]["average"]["total"])
        ca_away = float(team_stats_away["corners"]["against"]["average"]["total"])
        lam_total = (cf_home + ca_home + cf_away + ca_away) / 2.0
        return lam_total if lam_total > 0 else None
    except Exception:
        return None

def estimate_total_cards_lambda(team_stats_home: dict, team_stats_away: dict) -> Optional[float]:
    """
    Cards stats: API-Football teams/statistics suele tener 'cards' por rangos de minutos.
    Para MVP intentamos estimar un promedio total si existe.
    """
    try:
        cards_home = team_stats_home.get("cards", {})
        cards_away = team_stats_away.get("cards", {})
        # cards[("yellow"/"red")][minute_range]["total"]
        def sum_cards(cards_dict: dict) -> float:
            total = 0.0
            for color in ["yellow", "red"]:
                by_min = cards_dict.get(color, {})
                for k, v in by_min.items():
                    t = v.get("total")
                    if t is None:
                        continue
                    try:
                        total += float(t)
                    except:
                        pass
            return total

        # total temporada / partidos => necesitamos partidos jugados
        played_h = float(team_stats_home["fixtures"]["played"]["total"])
        played_a = float(team_stats_away["fixtures"]["played"]["total"])
        if played_h <= 0 or played_a <= 0:
            return None
        avg_cards_home = sum_cards(cards_home) / played_h
        avg_cards_away = sum_cards(cards_away) / played_a
        lam_total = avg_cards_home + avg_cards_away
        return lam_total if lam_total > 0 else None
    except Exception:
        return None


# =========================
# PICK EXTRACTION + p_model
# =========================
def extract_picks_from_event_odds(
    event_data: dict,
    sport_key: str,
    api: APIFootball,
    fixture_id_lookup: Optional[dict],
    team_stats_cache: Dict[Tuple[int,int,int], dict],
    league_id: Optional[int],
    season: Optional[int],
) -> List[Pick]:

    picks: List[Pick] = []

    ev_id = event_data.get("id")
    home = event_data.get("home_team")
    away = event_data.get("away_team")
    commence_time = event_data.get("commence_time")  # puede estar en event_odds response
    if not ev_id or not home or not away:
        return picks

    # 1) precio por market/outcome/point
    price_map: Dict[Tuple[str, str, Optional[float]], List[float]] = {}

    for bm in event_data.get("bookmakers", []):
        for m in bm.get("markets", []):
            mk = m.get("key")
            if not mk:
                continue
            for o in m.get("outcomes", []):
                name = o.get("name")
                price = o.get("price")
                point = o.get("point", None)
                if not name or not isinstance(price, (int, float)):
                    continue
                price_map.setdefault((mk, name, point), []).append(float(price))

    # 2) fixture match (API-Football) si disponible
    fixture = None
    home_team_id = None
    away_team_id = None
    if api.enabled() and fixture_id_lookup and league_id and season:
        fx = fixture_id_lookup.get((normalize_team(home), normalize_team(away)))
        if fx:
            fixture = fx
            try:
                home_team_id = fixture["teams"]["home"]["id"]
                away_team_id = fixture["teams"]["away"]["id"]
            except Exception:
                home_team_id = None
                away_team_id = None

    # 3) team stats para modelos
    th = None
    ta = None
    if api.enabled() and league_id and season and home_team_id and away_team_id:
        key_h = (league_id, season, home_team_id)
        key_a = (league_id, season, away_team_id)
        if key_h not in team_stats_cache:
            st = api.team_statistics(league_id, season, home_team_id)
            if st:
                team_stats_cache[key_h] = st
        if key_a not in team_stats_cache:
            st = api.team_statistics(league_id, season, away_team_id)
            if st:
                team_stats_cache[key_a] = st
        th = team_stats_cache.get(key_h)
        ta = team_stats_cache.get(key_a)

    lam_goals = None
    lam_corners = None
    lam_cards = None
    if th and ta:
        lam_goals = get_goal_lambdas(th, ta)  # (lam_home, lam_away)
        lam_corners = estimate_total_corners_lambda(th, ta)
        lam_cards = estimate_total_cards_lambda(th, ta)

    # 4) construir picks
    for (mk, name, point), prices in price_map.items():
        if not prices:
            continue
        best_odds = max(prices)
        med_odds = median(prices)

        if not (ODDS_MIN <= best_odds <= ODDS_MAX):
            continue
        if med_odds <= 1.01:
            continue
        if is_extreme_line(mk, point):
            continue

        p_market = 1.0 / med_odds

        # p_model (si se puede)
        p_model = None
        try:
            # GOALS total
            if mk == "alternate_totals" and point is not None and lam_goals:
                lam_total = lam_goals[0] + lam_goals[1]
                if name.lower().startswith("over"):
                    p_model = poisson_p_over(lam_total, float(point))
                elif name.lower().startswith("under"):
                    p_model = poisson_p_under(lam_total, float(point))

            # BTTS
            elif mk == "btts" and lam_goals:
                p_yes = btts_prob(lam_goals[0], lam_goals[1])
                if name.lower() == "yes":
                    p_model = p_yes
                elif name.lower() == "no":
                    p_model = 1.0 - p_yes

            # CORNERS total
            elif mk == "alternate_totals_corners" and point is not None and lam_corners:
                if name.lower().startswith("over"):
                    p_model = poisson_p_over(lam_corners, float(point))
                elif name.lower().startswith("under"):
                    p_model = poisson_p_under(lam_corners, float(point))

            # CARDS total
            elif mk == "alternate_totals_cards" and point is not None and lam_cards:
                if name.lower().startswith("over"):
                    p_model = poisson_p_over(lam_cards, float(point))
                elif name.lower().startswith("under"):
                    p_model = poisson_p_under(lam_cards, float(point))

        except Exception:
            p_model = None

        # p_final
        if p_model is not None:
            p_final = MODEL_WEIGHT * p_model + (1.0 - MODEL_WEIGHT) * p_market
        else:
            p_final = p_market

        if p_final <= 0:
            continue

        value_ratio = (best_odds / med_odds) - 1.0

        label = f"{home} vs {away} | {mk}: {name}"
        if point is not None:
            label += f" ({point})"
        label += f" @ {best_odds:.2f}"

        picks.append(
            Pick(
                match_id=ev_id,
                sport_key=sport_key,
                market=mk,
                outcome=name,
                point=point,
                odds_best=best_odds,
                odds_median=med_odds,
                p_market=p_market,
                p_model=p_model,
                p_final=p_final,
                value_ratio=value_ratio,
                label=label,
            )
        )

    # ranking base
    picks.sort(
        key=lambda p: (
            p.p_final
            + 0.10 * market_weight(p.market)
            + 0.10 * max(p.value_ratio, 0.0),
            p.odds_best
        ),
        reverse=True,
    )
    return picks


# =========================
# ACCA BUILDER
# =========================
def build_acca_closest(
    picks: List[Pick],
    target_odds: float,
    legs_range: Tuple[int, int],
    tol: float,
    mode: str,  # "cuoton" or "safe"
) -> List[Pick]:

    lo, hi = target_odds * (1 - tol), target_odds * (1 + tol)
    logT = math.log(target_odds)

    def pick_ok(p: Pick) -> bool:
        if mode == "safe":
            # singles/doubles "seguros" con EV+ aprox
            ev = p.p_final * p.odds_best - 1.0
            return (p.odds_best <= 2.60) and (p.p_final >= 0.42) and (ev > 0.02) and (p.value_ratio >= 0.02)

        # cuotón 900/300
        if p.p_final < MIN_P_LEG_CUOTON:
            return False
        if not (CUOTON_ODDS_MIN <= p.odds_best <= CUOTON_ODDS_MAX):
            return False
        if is_draw_like(p.market, p.outcome):
            return False

        # preferimos estos mercados
        allowed = set(ALL_EVENT_MARKETS)
        if p.market not in allowed:
            return False
        return True

    picks_use = [p for p in picks if pick_ok(p)]
    if not picks_use:
        return []

    def score(p: Pick) -> float:
        ev = p.p_final * p.odds_best - 1.0
        return (
            safe_log(p.p_final)
            + LAMBDA_ODDS * safe_log(p.odds_best)
            + 0.32 * market_weight(p.market)
            + 0.25 * max(p.value_ratio, 0.0)
            + 0.10 * max(ev, 0.0)
        )

    picks_sorted = sorted(picks_use, key=score, reverse=True)

    states = [([], 0.0, 0.0, set())]  # acca, log_odds, score, used_match_ids
    for p in picks_sorted:
        new_states = states[:]
        for acca, logod, sc, used in states:
            if p.match_id in used:
                continue
            if len(acca) >= legs_range[1]:
                continue
            acc2 = acca + [p]
            log2 = logod + math.log(p.odds_best)
            sc2 = sc + score(p)
            used2 = set(used); used2.add(p.match_id)
            new_states.append((acc2, log2, sc2, used2))

        new_states.sort(key=lambda x: (x[2] - TARGET_TIGHTNESS * abs(x[1] - logT)), reverse=True)
        states = new_states[:BEAM_WIDTH]

    # best in range
    best = []
    best_sc = -1e18
    for acca, logod, sc, _ in states:
        if not (legs_range[0] <= len(acca) <= legs_range[1]):
            continue
        odds_total = math.exp(logod)
        if lo <= odds_total <= hi and sc > best_sc:
            best, best_sc = acca, sc
    if best:
        return best

    # closest
    closest = []
    closest_dist = 1e18
    closest_sc = -1e18
    for acca, logod, sc, _ in states:
        if not (legs_range[0] <= len(acca) <= legs_range[1]):
            continue
        dist = abs(logod - logT)
        if dist < closest_dist or (abs(dist - closest_dist) < 1e-9 and sc > closest_sc):
            closest, closest_dist, closest_sc = acca, dist, sc
    return closest


# =========================
# FIXTURE LOOKUP (API-Football)
# =========================
def build_fixture_lookup(
    api: APIFootball,
    sport_key: str,
    events: List[dict],
    league_id: int,
    season: int,
) -> Dict[Tuple[str, str], dict]:
    """
    Crea mapa (home_norm, away_norm) -> fixture dict
    usando fixtures por fecha.
    """
    if not api.enabled():
        return {}

    # fechas únicas de commence_time
    dates = set()
    for ev in events:
        ct = ev.get("commence_time")
        if not ct:
            continue
        try:
            dt = datetime.datetime.fromisoformat(ct.replace("Z", "+00:00"))
            dates.add(dt.date().isoformat())
        except Exception:
            continue

    lookup = {}
    for d in sorted(dates):
        fixtures = api.fixtures_by_date_league(d, league_id, season)
        for fx in fixtures:
            try:
                h = fx["teams"]["home"]["name"]
                a = fx["teams"]["away"]["name"]
            except Exception:
                continue
            lookup[(normalize_team(h), normalize_team(a))] = fx

    # fallback fuzzy: si Odds API usa nombres distintos
    # solo aplicamos cuando no encuentre exacto
    def find_best(home: str, away: str) -> Optional[dict]:
        best = None
        best_score = 0.0
        hn = normalize_team(home)
        an = normalize_team(away)
        for (h2, a2), fx in lookup.items():
            sc = 0.5 * fuzzy_match(hn, h2) + 0.5 * fuzzy_match(an, a2)
            if sc > best_score:
                best_score, best = sc, fx
        return best if best_score >= 0.82 else None

    fixed = {}
    for ev in events:
        h = ev.get("home_team")
        a = ev.get("away_team")
        if not h or not a:
            continue
        key = (normalize_team(h), normalize_team(a))
        if key in lookup:
            fixed[key] = lookup[key]
        else:
            fx = find_best(h, a)
            if fx:
                fixed[key] = fx

    return fixed


# =========================
# MAIN
# =========================
def main() -> None:
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tg_send(f"🚀 RUN {now} sha={GITHUB_SHA} primary={REGIONS_PRIMARY} fallback={REGIONS_FALLBACK}")

    api = APIFootball()
    if api.enabled():
        st = api.status()
        tg_send(f"🧠 API-Football ON mode={api.mode} status_ok={'yes' if st else 'no'}")
    else:
        tg_send("🧠 API-Football OFF (no key). p_model será None.")

    # 1) List events
    all_events: List[Tuple[str, dict]] = []
    for sk in SPORT_KEYS:
        data = fetch_odds_list(sk, REGIONS_PRIMARY)
        for ev in data:
            all_events.append((sk, ev))

    if not all_events:
        tg_send("⚠️ No llegaron eventos desde /odds.")
        return

    # 2) Choose events to enrich
    all_events.sort(key=lambda t: fav_strength_from_h2h(t[1]), reverse=True)
    to_enrich = all_events[:MAX_EVENTS_ENRICH]
    tg_send(f"🔎 Enriqueciendo {len(to_enrich)} eventos con corners/cards-first + core…")

    # 3) Build fixture lookups per league/day (minimizando llamadas)
    team_stats_cache: Dict[Tuple[int,int,int], dict] = {}
    fixture_lookups: Dict[str, Dict[Tuple[str,str], dict]] = {}

    # agrupamos por sport_key para construir lookup 1 vez
    for sk in SPORT_KEYS:
        league_id = LEAGUE_MAP.get(sk)
        if not league_id:
            continue
        # eventos de ese sport_key dentro de to_enrich
        evs = [ev for s, ev in to_enrich if s == sk]
        if not evs:
            continue
        season = season_for_date(datetime.datetime.utcnow())
        if api.enabled():
            fixture_lookups[sk] = build_fixture_lookup(api, sk, evs, league_id, season)

    # 4) Enrich + extract picks
    picks: List[Pick] = []
    markets_seen_global = set()
    corners_cards_seen = set()

    for sk, ev in to_enrich:
        ev_id = ev.get("id")
        if not ev_id:
            continue

        data, mk_seen = fetch_event_odds_best_effort(sk, ev_id)
        if not data:
            continue

        markets_seen_global |= mk_seen
        for k in mk_seen:
            if k in CORNERS_CARDS_MARKETS:
                corners_cards_seen.add(k)

        # attach missing event fields for extractor (commence_time etc.)
        # si el endpoint event_odds no trae commence_time, lo tomamos del listado:
        if "commence_time" not in data and ev.get("commence_time"):
            data["commence_time"] = ev["commence_time"]
        if "home_team" not in data:
            data["home_team"] = ev.get("home_team")
        if "away_team" not in data:
            data["away_team"] = ev.get("away_team")

        league_id = LEAGUE_MAP.get(sk)
        season = season_for_date(datetime.datetime.utcnow())
        lookup = fixture_lookups.get(sk)

        picks.extend(
            extract_picks_from_event_odds(
                event_data=data,
                sport_key=sk,
                api=api,
                fixture_id_lookup=lookup,
                team_stats_cache=team_stats_cache,
                league_id=league_id,
                season=season,
            )
        )

    if not picks:
        tg_send("⚠️ No se extrajeron picks.")
        return

    tg_send(
        f"📦 Picks={len(picks)} | markets_global={sorted(list(markets_seen_global))}\n"
        f"🎯 corners/cards present={sorted(list(corners_cards_seen)) if corners_cards_seen else 'NO'}"
    )

    # 5) TOP candidatos (con p_final y p_model)
    top10 = sorted(
        picks,
        key=lambda p: (p.p_final + 0.12*market_weight(p.market) + 0.10*max(p.value_ratio,0.0)),
        reverse=True
    )[:10]

    lines = []
    for i, p in enumerate(top10, 1):
        pm = f"{p.p_model*100:.1f}%" if p.p_model is not None else "NA"
        lines.append(f"{i}) {p.label} | p_final~{p.p_final*100:.1f}% | p_model={pm} | value~{p.value_ratio*100:.1f}%")
    tg_send("✅ TOP CANDIDATOS (p_final)\n" + "\n".join(lines))

    # 6) Cuotón ~900
    acca900 = build_acca_closest(picks, TARGET_900, LEGS_900, ODDS_TOL_900, mode="cuoton")
    if acca900:
        total = odds_product(acca900)
        lo, hi = TARGET_900*(1-ODDS_TOL_900), TARGET_900*(1+ODDS_TOL_900)
        inside = lo <= total <= hi
        msg = [f"🎯 CUOTÓN ~900 ({'EN RANGO' if inside else 'MÁS CERCANO'})",
               f"Patas: {len(acca900)} | Cuota: {total:.2f}"]
        for idx, p in enumerate(acca900, 1):
            pm = f"{p.p_model*100:.1f}%" if p.p_model is not None else "NA"
            msg.append(f"{idx}) {p.label} | p_final~{p.p_final*100:.1f}% | p_model={pm}")
        tg_send("\n".join(msg))
    else:
        tg_send("⚠️ No pude construir cuotón ~900 con filtros actuales.")

    # 7) Cuotón ~300
    acca300 = build_acca_closest(picks, TARGET_300, LEGS_300, ODDS_TOL_300, mode="cuoton")
    if acca300:
        total = odds_product(acca300)
        lo, hi = TARGET_300*(1-ODDS_TOL_300), TARGET_300*(1+ODDS_TOL_300)
        inside = lo <= total <= hi
        msg = [f"🟦 CUOTÓN ~300 ({'EN RANGO' if inside else 'MÁS CERCANO'})",
               f"Patas: {len(acca300)} | Cuota: {total:.2f}"]
        for idx, p in enumerate(acca300, 1):
            pm = f"{p.p_model*100:.1f}%" if p.p_model is not None else "NA"
            msg.append(f"{idx}) {p.label} | p_final~{p.p_final*100:.1f}% | p_model={pm}")
        tg_send("\n".join(msg))

    # 8) Modo Seguro (singles + dobles)
    safe = [p for p in picks if (p.p_final*p.odds_best - 1.0) > 0.02 and p.p_final >= 0.42 and p.odds_best <= 2.60]
    safe.sort(key=lambda p: (p.p_final*p.odds_best - 1.0), reverse=True)
    top_singles = []
    used_matches = set()
    for p in safe:
        if p.match_id in used_matches:
            continue
        used_matches.add(p.match_id)
        top_singles.append(p)
        if len(top_singles) >= 6:
            break

    if top_singles:
        msg = ["🛡️ MODO SEGURO (SINGLES EV+ aprox)"]
        for i, p in enumerate(top_singles, 1):
            ev = p.p_final*p.odds_best - 1.0
            pm = f"{p.p_model*100:.1f}%" if p.p_model is not None else "NA"
            msg.append(f"{i}) {p.label} | p_final~{p.p_final*100:.1f}% | EV~{ev*100:.1f}% | p_model={pm}")
        tg_send("\n".join(msg))

        doubles = []
        for i in range(len(top_singles)):
            for j in range(i+1, len(top_singles)):
                a, b = top_singles[i], top_singles[j]
                if a.match_id == b.match_id:
                    continue
                odds2 = a.odds_best * b.odds_best
                p2 = a.p_final * b.p_final
                ev2 = p2 * odds2 - 1.0
                doubles.append((ev2, a, b, odds2, p2))
        doubles.sort(key=lambda x: x[0], reverse=True)
        best2 = doubles[:3]
        if best2:
            msg = ["🛡️ MODO SEGURO (DOBLES EV+ aprox)"]
            for k, (ev2, a, b, odds2, p2) in enumerate(best2, 1):
                msg.append(f"Double {k} | cuota~{odds2:.2f} | p~{p2*100:.2f}% | EV~{ev2*100:.1f}%")
                msg.append(f"- {a.label}")
                msg.append(f"- {b.label}")
                msg.append("")
            tg_send("\n".join(msg).strip())


if __name__ == "__main__":
    main()
