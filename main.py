import os
import math
import time
import requests
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =========================
# ENV / CONFIG
# =========================
ODDS_API_KEY = os.environ["ODDS_API_KEY"]
TG_BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]
GITHUB_SHA = os.environ.get("GITHUB_SHA", "")[:7]

REGIONS = "uk"          # suele ir bien para soccer en The Odds API
ODDS_FORMAT = "decimal"

# Ligas (baja o sube según créditos)
SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
]

# Objetivos
TARGET_900 = 900.0
TARGET_300 = 300.0

# Para replicar “cuotón borroso”: 4–6 patas, cuotas altas por pata
LEGS_RANGE_900 = (4, 6)
LEGS_RANGE_300 = (4, 6)

# Tolerancia de cuota final
ODDS_TOL_900 = 0.30   # ±30%
ODDS_TOL_300 = 0.25

# Beam search
BEAM_WIDTH = 800

# Picks candidatos (por pata)
ODDS_MIN_GENERAL = 1.50
ODDS_MAX_GENERAL = 15.0

# “Estilo borroso” (patas altas para el 900)
ODDS_MIN_BORROSO = 4.0
ODDS_MAX_BORROSO = 9.0

# Cuántos eventos enriquecemos con /events/{eventId}/odds para no gastar créditos
MAX_EVENTS_ENRICH = 12

# Mercados adicionales soportados (según la lista oficial de markets)
# OJO: no metas market keys inventados (ej: double_chance) porque da INVALID_MARKET
EXTRA_MARKETS = [
    "alternate_spreads",
    "alternate_totals",
    "btts",
    "draw_no_bet",
    "h2h_3_way",
    "team_totals",
    "alternate_team_totals",
]

# Para que el combinador no se vaya a puros favoritos 1.60 y nunca llegue a 900
LAMBDA_ODDS = 0.40

# Pausa mínima entre llamadas event_odds (por estabilidad)
EVENT_ODDS_SLEEP_SEC = 0.25


# =========================
# MODELS / DATA
# =========================
@dataclass(frozen=True)
class Pick:
    match_id: str
    sport_key: str
    label: str
    odds_best: float
    odds_median: float
    p_est: float          # p estimada simple desde odds_median
    value_ratio: float    # (odds_best / odds_median - 1)


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
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))


# =========================
# ODDS API CALLS
# =========================
def fetch_odds_list(sport_key: str) -> list:
    """
    Cheap listing: only h2h, returns events + bookmakers.
    """
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
    """
    Enriched: one event, multiple markets.
    """
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
    last = r.headers.get("x-requests-last")  # cuánto costó esta llamada (si aplica)

    if r.status_code != 200:
        tg_send(f"❌ event_odds ERROR {sport_key} {event_id} HTTP {r.status_code} last={last} used={used} rem={rem}\n{r.text[:300]}")
        return None

    data = r.json()

    mk = set()
    for bm in data.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m.get("key"):
                mk.add(m.get("key"))

    tg_send(f"🧩 event_odds OK {sport_key} markets_returned={sorted(mk)[:12]} last={last} used={used} rem={rem}")
    return data


# =========================
# EVENT SCORING (to choose which events to enrich)
# =========================
def fav_strength_from_h2h(event: dict) -> float:
    """
    Proxy: stronger favorite => more likely to have interesting alternate_spreads etc.
    Returns approx favorite probability using min odds found.
    """
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
    """
    Parse all returned markets/outcomes.
    Build price distributions per (market_key, outcome_name, point) across bookmakers.
    """
    picks: List[Pick] = []
    ev_id = event_data.get("id")
    home = event_data.get("home_team")
    away = event_data.get("away_team")
    if not ev_id or not home or not away:
        return picks

    # key = (market_key, outcome_name, point)
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
                k = (mk, name, point)
                price_map.setdefault(k, []).append(float(price))

    for (mk, name, point), prices in price_map.items():
        if not prices:
            continue
        best_odds = max(prices)
        med_odds = median(prices)

        if not (ODDS_MIN_GENERAL <= best_odds <= ODDS_MAX_GENERAL):
            continue
        if med_odds <= 1.01:
            continue

        p_est = 1.0 / med_odds
        value_ratio = (best_odds / med_odds) - 1.0

        if point is None:
            label = f"{home} vs {away} | {mk}: {name} @ {best_odds:.2f}"
        else:
            label = f"{home} vs {away} | {mk}: {name} ({point}) @ {best_odds:.2f}"

        picks.append(
            Pick(
                match_id=ev_id,
                sport_key=sport_key,
                label=label,
                odds_best=best_odds,
                odds_median=med_odds,
                p_est=p_est,
                value_ratio=value_ratio,
            )
        )

    # Orden para que primero estén “más probables” y “mejor value”
    picks.sort(key=lambda p: (p.p_est, p.value_ratio, p.odds_best), reverse=True)
    return picks


# =========================
# ACCA BUILDER
# =========================
def build_acca_closest(
    picks: List[Pick],
    target_odds: float,
    legs_range: Tuple[int, int],
    tol: float,
    prefer_borroso: bool,
) -> List[Pick]:
    """
    Build best acca near target. If none in range, returns closest.
    One pick per match_id.
    """

    # Optional: “estilo borroso” = patas altas
    if prefer_borroso:
        borroso = [p for p in picks if ODDS_MIN_BORROSO <= p.odds_best <= ODDS_MAX_BORROSO]
        # si hay pocos, usar general
        if len(borroso) >= 25:
            picks_use = borroso
        else:
            picks_use = picks
    else:
        picks_use = picks

    lo, hi = target_odds * (1 - tol), target_odds * (1 + tol)
    logT = math.log(target_odds)

    # Ranking base para iterar picks “buenos primero”
    # Empuje de odds para poder llegar a 900 con 4–6 patas
    def base_rank(p: Pick) -> float:
        return safe_log(p.p_est) + 0.15 * safe_log(p.odds_best) + 0.35 * max(p.value_ratio, 0.0)

    picks_sorted = sorted(picks_use, key=base_rank, reverse=True)

    # Beam search states: (acca, log_odds_total, score_total, used_match_ids)
    states = [([], 0.0, 0.0, set())]

    for p in picks_sorted:
        new_states = states[:]
        for acca, logod, sc, used in states:
            if p.match_id in used:
                continue
            if len(acca) >= legs_range[1]:
                continue

            acc2 = acca + [p]
            log2 = logod + math.log(p.odds_best)

            # Score: prob + push for higher odds + slight bonus for value
            sc2 = sc + safe_log(p.p_est) + LAMBDA_ODDS * safe_log(p.odds_best) + 0.25 * max(p.value_ratio, 0.0)

            used2 = set(used)
            used2.add(p.match_id)
            new_states.append((acc2, log2, sc2, used2))

        # keep best states combining: score - closeness penalty
        new_states.sort(key=lambda x: (x[2] - 0.55 * abs(x[1] - logT)), reverse=True)
        states = new_states[:BEAM_WIDTH]

    # 1) best within tolerance window
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

    # 2) else closest
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


def odds_product(acca: List[Pick]) -> float:
    prod = 1.0
    for p in acca:
        prod *= p.odds_best
    return prod


# =========================
# MAIN
# =========================
def main() -> None:
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tg_send(f"🚀 RUN {now} sha={GITHUB_SHA} regions={REGIONS}")

    # 1) List events cheaply
    all_events: List[Tuple[str, dict]] = []
    for sk in SPORT_KEYS:
        data = fetch_odds_list(sk)
        for ev in data:
            all_events.append((sk, ev))

    if not all_events:
        tg_send("⚠️ No llegaron eventos desde /odds. Revisa sport_keys o región.")
        return

    # 2) Choose events to enrich
    all_events.sort(key=lambda t: fav_strength_from_h2h(t[1]), reverse=True)
    to_enrich = all_events[:MAX_EVENTS_ENRICH]
    tg_send(f"🔎 Enriqueciendo {len(to_enrich)} eventos con mercados extra…")

    # 3) Enrich + extract picks
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

        # track markets returned (debug)
        for bm in enriched.get("bookmakers", []):
            for m in bm.get("markets", []):
                if m.get("key"):
                    markets_seen_global.add(m.get("key"))

        picks.extend(extract_picks_from_event_odds(enriched, sk))

    if not picks:
        tg_send(
            "⚠️ No se extrajeron picks de mercados extra.\n"
            "Esto suele significar: esos markets no están disponibles para soccer en tu región/bookmakers con este feed.\n"
            f"Markets pedidos: {EXTRA_MARKETS}"
        )
        return

    tg_send(f"📦 Total picks extraídos: {len(picks)} | markets_returned_global={sorted(markets_seen_global)[:12]}")

    # 4) TOP candidates to verify
    top10 = picks[:10]
    msg_top = "✅ TOP CANDIDATOS (mercados extra)\n" + "\n".join(
        [f"{i+1}) {p.label} | p~{p.p_est*100:.1f}% | value~{p.value_ratio*100:.1f}%"
         for i, p in enumerate(top10)]
    )
    tg_send(msg_top)

    # 5) Build cuotón 900 (prefer borroso style)
    acca_900 = build_acca_closest(
        picks=picks,
        target_odds=TARGET_900,
        legs_range=LEGS_RANGE_900,
        tol=ODDS_TOL_900,
        prefer_borroso=True,
    )
    if acca_900:
        total_900 = odds_product(acca_900)
        lo900, hi900 = TARGET_900*(1-ODDS_TOL_900), TARGET_900*(1+ODDS_TOL_900)
        inside = (lo900 <= total_900 <= hi900)

        lines = "\n".join([f"{i+1}) {p.label}" for i, p in enumerate(acca_900)])
        tg_send(
            f"🎯 CUOTÓN ~900 ({'EN RANGO' if inside else 'MÁS CERCANO'})\n"
            f"Patas: {len(acca_900)} | Cuota: {total_900:.2f}\n\n{lines}"
        )
    else:
        tg_send("⚠️ No se pudo construir cuotón 900 (algo raro).")

    # 6) Build cuotón 300 (más flexible)
    acca_300 = build_acca_closest(
        picks=picks,
        target_odds=TARGET_300,
        legs_range=LEGS_RANGE_300,
        tol=ODDS_TOL_300,
        prefer_borroso=False,
    )
    if acca_300:
        total_300 = odds_product(acca_300)
        lo300, hi300 = TARGET_300*(1-ODDS_TOL_300), TARGET_300*(1+ODDS_TOL_300)
        inside = (lo300 <= total_300 <= hi300)

        lines = "\n".join([f"{i+1}) {p.label}" for i, p in enumerate(acca_300)])
        tg_send(
            f"🟦 CUOTÓN ~300 ({'EN RANGO' if inside else 'MÁS CERCANO'})\n"
            f"Patas: {len(acca_300)} | Cuota: {total_300:.2f}\n\n{lines}"
        )

    # 7) Value picks (muy simple): mejores prices vs mediana
    # (No es EV real todavía; es “value vs consenso”)
    value_sorted = sorted(picks, key=lambda p: p.value_ratio, reverse=True)
    value_top = [p for p in value_sorted if p.value_ratio >= 0.06][:5]  # >= +6%
    if value_top:
        lines = "\n".join(
            [f"{i+1}) {p.label} | med~{p.odds_median:.2f} | value~{p.value_ratio*100:.1f}%"
             for i, p in enumerate(value_top)]
        )
        tg_send("💎 VALUE PICKS (vs consenso)\n" + lines)


if __name__ == "__main__":
    main()




