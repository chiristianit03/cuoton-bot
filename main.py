import os
import math
import time
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

REGIONS = "uk"
ODDS_FORMAT = "decimal"

SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
]

# =========================
# OBJETIVOS
# =========================
TARGET_900 = 900.0
TARGET_300 = 300.0

LEGS_900 = (4, 6)
LEGS_300 = (4, 6)

# Tolerancia: si quieres que "900" sea más estricto, baja esto.
ODDS_TOL_900 = 0.18   # ~738 a 1062
ODDS_TOL_300 = 0.22

BEAM_WIDTH = 900

# Cuotas por pata (global)
ODDS_MIN = 1.30
ODDS_MAX = 25.0

# Para "cuotón estilo borroso": cuotas altas pero no absurdas
CUOTON_ODDS_MIN = 3.0
CUOTON_ODDS_MAX = 7.5

# Mínimo de probabilidad por pata (estimada por consenso)
# (sube esto si quieres más "probable" aunque baje la cuota total)
MIN_P_LEG_CUOTON = 0.20  # 20%

# Limitamos cuántos eventos enriquecemos para no gastar créditos
MAX_EVENTS_ENRICH = 10
EVENT_ODDS_SLEEP_SEC = 0.20

# IMPORTANT: keys oficiales del propio The Odds API
# Incluye corners/cards/double chance además de alt totals/spreads
EXTRA_MARKETS = [
    "alternate_spreads",
    "alternate_totals",
    "btts",
    "draw_no_bet",
    "h2h_3_way",
    "team_totals",
    "alternate_team_totals",
    "alternate_spreads_corners",
    "alternate_totals_corners",
    "alternate_spreads_cards",
    "alternate_totals_cards",
    "double_chance",
]

# Empuje a odds para poder llegar a 900 con 4–6 patas
LAMBDA_ODDS = 0.35
# Qué tan obsesivo es el beam con acercarse al target (más = más cerca)
TARGET_TIGHTNESS = 1.35


# =========================
# DATA MODEL
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
    p_est: float          # ~ 1 / odds_median
    value_ratio: float    # odds_best / odds_median - 1
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


# =========================
# MARKET PREFERENCES / FILTERS
# =========================
def market_weight(mk: str) -> float:
    """
    Preferimos mercados más "tipo fórmula" (alt lines, corners, cards)
    y penalizamos cosas tipo draw/random.
    """
    return {
        "alternate_totals_corners": 0.25,
        "alternate_spreads_corners": 0.22,
        "alternate_totals_cards": 0.22,
        "alternate_spreads_cards": 0.20,
        "alternate_totals": 0.20,
        "alternate_spreads": 0.18,
        "btts": 0.14,
        "team_totals": 0.12,
        "alternate_team_totals": 0.12,
        "draw_no_bet": 0.08,
        "double_chance": 0.06,
        "h2h_3_way": -0.06,   # lo dejamos pero con penalización
    }.get(mk, 0.0)

def is_extreme_line(mk: str, outcome: str, point: Optional[float]) -> bool:
    """
    Evita líneas "suicidas" típicas (ej: Under 0.5 goles a 12, etc.)
    Ajusta rangos a tu gusto.
    """
    if point is None:
        return False

    # Goals totals
    if mk in ("alternate_totals", "totals"):
        # evitamos 0.5 y 6.5+ como líneas muy extremas
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

    # Spreads (AH)
    if mk == "alternate_spreads":
        # típico AH útil: -2.5 a +2.5
        if point < -2.5 or point > 2.5:
            return True

    # Corner/Card handicaps
    if mk in ("alternate_spreads_corners", "alternate_spreads_cards"):
        if point < -4.5 or point > 4.5:
            return True

    return False

def is_draw_like(mk: str, outcome: str) -> bool:
    # Evita draws para cuotón (normalmente baja mucho la prob)
    return mk == "h2h_3_way" and outcome.lower() == "draw"


# =========================
# ODDS API
# =========================
def fetch_odds_list(sport_key: str) -> list:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": "h2h",
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    used = r.headers.get("x-requests-used")
    rem = r.headers.get("x-requests-remaining")

    if r.status_code != 200:
        tg_send(f"❌ /odds ERROR {sport_key} HTTP {r.status_code} used={used} rem={rem}\n{r.text[:300]}")
        return []

    data = r.json()
    bm0 = len(data[0].get("bookmakers", [])) if data else 0
    tg_send(f"✅ /odds {sport_key}: events={len(data)} first_bookmakers={bm0} used={used} rem={rem} sha={GITHUB_SHA}")
    return data

def fetch_event_odds(sport_key: str, event_id: str, markets: List[str]) -> Optional[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=30)
    used = r.headers.get("x-requests-used")
    rem = r.headers.get("x-requests-remaining")
    last = r.headers.get("x-requests-last")

    if r.status_code != 200:
        tg_send(f"❌ event_odds ERROR {sport_key} {event_id} HTTP {r.status_code} last={last} used={used} rem={rem}\n{r.text[:300]}")
        return None

    data = r.json()

    mk = set()
    for bm in data.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m.get("key"):
                mk.add(m.get("key"))

    tg_send(f"🧩 event_odds OK {sport_key} markets_returned={sorted(mk)[:16]} last={last} used={used} rem={rem}")
    return data

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
# PICK EXTRACTION
# =========================
def extract_picks_from_event_odds(event_data: dict, sport_key: str) -> List[Pick]:
    picks: List[Pick] = []

    ev_id = event_data.get("id")
    home = event_data.get("home_team")
    away = event_data.get("away_team")
    if not ev_id or not home or not away:
        return picks

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

    for (mk, name, point), prices in price_map.items():
        if not prices:
            continue

        best_odds = max(prices)
        med_odds = median(prices)

        if not (ODDS_MIN <= best_odds <= ODDS_MAX):
            continue
        if med_odds <= 1.01:
            continue

        # filtros de sentido común
        if is_extreme_line(mk, name, point):
            continue

        p_est = 1.0 / med_odds
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
                p_est=p_est,
                value_ratio=value_ratio,
                label=label,
            )
        )

    # Orden inicial (más probable + preferencia de mercado + value)
    picks.sort(
        key=lambda p: (
            p.p_est
            + 0.08 * market_weight(p.market)
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

    def pick_ok_for_mode(p: Pick) -> bool:
        if mode == "safe":
            # safe: evitamos odds muy altas
            return p.odds_best <= 2.60 and p.p_est >= 0.40 and p.value_ratio >= 0.03
        # cuoton:
        if p.p_est < MIN_P_LEG_CUOTON:
            return False
        if not (CUOTON_ODDS_MIN <= p.odds_best <= CUOTON_ODDS_MAX):
            return False
        if is_draw_like(p.market, p.outcome):
            return False
        # Para cuotón, preferimos mercados tipo fórmula:
        if p.market not in (
            "alternate_totals",
            "alternate_spreads",
            "alternate_totals_corners",
            "alternate_spreads_corners",
            "alternate_totals_cards",
            "alternate_spreads_cards",
            "btts",
            "draw_no_bet",
            "double_chance",
            "team_totals",
            "alternate_team_totals",
            "h2h_3_way",
        ):
            return False
        return True

    picks_use = [p for p in picks if pick_ok_for_mode(p)]
    if not picks_use:
        return []

    def score(p: Pick) -> float:
        # log prob + push de odds + preferencia de mercado + value
        return (
            safe_log(p.p_est)
            + LAMBDA_ODDS * safe_log(p.odds_best)
            + 0.30 * market_weight(p.market)
            + 0.25 * max(p.value_ratio, 0.0)
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

        # score - penalty(distance to target)
        new_states.sort(key=lambda x: (x[2] - TARGET_TIGHTNESS * abs(x[1] - logT)), reverse=True)
        states = new_states[:BEAM_WIDTH]

    # 1) best in range
    best: List[Pick] = []
    best_sc = -1e18
    for acca, logod, sc, _ in states:
        if not (legs_range[0] <= len(acca) <= legs_range[1]):
            continue
        odds_total = math.exp(logod)
        if lo <= odds_total <= hi and sc > best_sc:
            best, best_sc = acca, sc
    if best:
        return best

    # 2) closest
    closest: List[Pick] = []
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
# MAIN
# =========================
def main() -> None:
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tg_send(f"🚀 RUN {now} sha={GITHUB_SHA} regions={REGIONS}")

    # 1) list events
    all_events: List[Tuple[str, dict]] = []
    for sk in SPORT_KEYS:
        data = fetch_odds_list(sk)
        for ev in data:
            all_events.append((sk, ev))

    if not all_events:
        tg_send("⚠️ No llegaron eventos desde /odds.")
        return

    # 2) choose events to enrich (favoritos fuertes suelen tener más líneas)
    all_events.sort(key=lambda t: fav_strength_from_h2h(t[1]), reverse=True)
    to_enrich = all_events[:MAX_EVENTS_ENRICH]
    tg_send(f"🔎 Enriqueciendo {len(to_enrich)} eventos con mercados extra…")

    picks: List[Pick] = []
    markets_seen_global = set()

    for sk, ev in to_enrich:
        ev_id = ev.get("id")
        if not ev_id:
            continue

        enriched = fetch_event_odds(sk, ev_id, EXTRA_MARKETS)
        time.sleep(EVENT_ODDS_SLEEP_SEC)
        if not enriched:
            continue

        for bm in enriched.get("bookmakers", []):
            for m in bm.get("markets", []):
                if m.get("key"):
                    markets_seen_global.add(m.get("key"))

        picks.extend(extract_picks_from_event_odds(enriched, sk))

    if not picks:
        tg_send(
            "⚠️ No se extrajeron picks.\n"
            "Si corners/cards no aparecen, puede ser cobertura limitada del feed/bookmakers.\n"
        )
        return

    tg_send(f"📦 Picks: {len(picks)} | markets_returned_global={sorted(markets_seen_global)}")

    # 3) Top candidates (para ver si ya aparecen corners/cards)
    top10 = sorted(picks, key=lambda p: (p.p_est + 0.08*market_weight(p.market)), reverse=True)[:10]
    tg_send(
        "✅ TOP CANDIDATOS\n" + "\n".join(
            [f"{i+1}) {p.label} | p~{p.p_est*100:.1f}% | value~{p.value_ratio*100:.1f}%"
             for i, p in enumerate(top10)]
        )
    )

    # 4) Cuotón 900
    acca900 = build_acca_closest(picks, TARGET_900, LEGS_900, ODDS_TOL_900, mode="cuoton")
    if acca900:
        total = odds_product(acca900)
        lo, hi = TARGET_900*(1-ODDS_TOL_900), TARGET_900*(1+ODDS_TOL_900)
        inside = lo <= total <= hi
        tg_send(
            f"🎯 CUOTÓN ~900 ({'EN RANGO' if inside else 'MÁS CERCANO'})\n"
            f"Patas: {len(acca900)} | Cuota: {total:.2f}\n\n" +
            "\n".join([f"{i+1}) {p.label}" for i, p in enumerate(acca900)])
        )
    else:
        tg_send("⚠️ No pude construir cuotón ~900 con filtros actuales (sube MAX_EVENTS_ENRICH o ajusta rangos).")

    # 5) Cuotón 300
    acca300 = build_acca_closest(picks, TARGET_300, LEGS_300, ODDS_TOL_300, mode="cuoton")
    if acca300:
        total = odds_product(acca300)
        lo, hi = TARGET_300*(1-ODDS_TOL_300), TARGET_300*(1+ODDS_TOL_300)
        inside = lo <= total <= hi
        tg_send(
            f"🟦 CUOTÓN ~300 ({'EN RANGO' if inside else 'MÁS CERCANO'})\n"
            f"Patas: {len(acca300)} | Cuota: {total:.2f}\n\n" +
            "\n".join([f"{i+1}) {p.label}" for i, p in enumerate(acca300)])
        )

    # 6) Modo Seguro (singles EV+ y dobles)
    # EV aproximado: EV ~ p_est * odds_best - 1 (p_est de mediana)
    safe_candidates = []
    for p in picks:
        ev = p.p_est * p.odds_best - 1.0
        if p.odds_best <= 2.60 and p.p_est >= 0.40 and ev > 0.02:
            safe_candidates.append((ev, p))

    safe_candidates.sort(key=lambda x: x[0], reverse=True)
    top_singles = [p for _, p in safe_candidates[:6]]

    if top_singles:
        tg_send(
            "🛡️ MODO SEGURO (SINGLES EV+ aprox)\n" +
            "\n".join(
                [f"{i+1}) {p.label} | p~{p.p_est*100:.1f}% | EV~{(p.p_est*p.odds_best-1)*100:.1f}%"
                 for i, p in enumerate(top_singles)]
            )
        )

        # dobles: dos partidos distintos
        doubles = []
        for i in range(len(top_singles)):
            for j in range(i+1, len(top_singles)):
                a, b = top_singles[i], top_singles[j]
                if a.match_id == b.match_id:
                    continue
                odds2 = a.odds_best * b.odds_best
                p2 = a.p_est * b.p_est
                ev2 = p2 * odds2 - 1.0
                doubles.append((ev2, a, b, odds2, p2))
        doubles.sort(key=lambda x: x[0], reverse=True)
        best2 = doubles[:3]
        if best2:
            tg_send(
                "🛡️ MODO SEGURO (DOBLES EV+ aprox)\n" +
                "\n\n".join(
                    [f"Double {k+1} | cuota~{odds2:.2f} | p~{p2*100:.2f}% | EV~{ev2*100:.1f}%\n- {a.label}\n- {b.label}"
                     for k, (ev2, a, b, odds2, p2) in enumerate(best2)]
                )
            )


if __name__ == "__main__":
    main()




