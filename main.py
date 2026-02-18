import os, math, requests
from dataclasses import dataclass
from typing import List, Tuple

ODDS_API_KEY = os.environ["ODDS_API_KEY"]
TG_BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]

# Ligas iniciales (puedes ampliar)
SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
]

REGION = "uk"              # en soccer suele venir mejor que eu
MARKETS = "h2h"            # lo seguro en v4 para fútbol
ODDS_PER_LEG_RANGE = (2.0, 12.0)   # baja el mínimo para que encuentre picks


TARGET_ODDS = 900.0
LEGS_RANGE = (4, 6)
ODDS_PER_LEG_RANGE = (3.5, 10.0)

@dataclass(frozen=True)
class Pick:
    match_id: str
    label: str
    odds: float
    p_est: float  # estimación simple desde consenso
    sport_key: str

def tg_send(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg})

def median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid-1] + xs[mid]) / 2

def implied_prob_from_odds(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 0.0

def fetch_odds(sport_key: str) -> list:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def extract_candidates(data: list, sport_key: str) -> List[Pick]:
    picks: List[Pick] = []
    for ev in data:
        match_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not match_id or not home or not away: 
            continue

        # Recorremos bookmakers -> markets -> outcomes
        # Estimación simple: para cada outcome tomamos mediana de (1/odds) entre bookmakers.
        outcome_odds_map = {}  # key: (market_key, outcome_name, point) -> list_odds
        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                mk = m.get("key")
                for o in m.get("outcomes", []):
                    name = o.get("name")
                    price = o.get("price")
                    point = o.get("point", None)
                    if not name or not isinstance(price, (int, float)):
                        continue
                    k = (mk, name, point)
                    outcome_odds_map.setdefault(k, []).append(float(price))

        for (mk, name, point), odds_list in outcome_odds_map.items():
            # Elegimos odds “atractivas” para cuotón (max odds disponible),
            # y prob estimada por consenso (mediana de probs).
            max_odds = max(odds_list)
            if not (ODDS_PER_LEG_RANGE[0] <= max_odds <= ODDS_PER_LEG_RANGE[1]):
                continue

            probs = [implied_prob_from_odds(x) for x in odds_list]
            p_med = median(probs)

            # Filtro básico: evita locuras con prob demasiado baja
            if p_med < 0.08:  # ~12.5x o más; ajusta luego
                continue

            point_txt = f" {point}" if point is not None else ""
            label = f"{home} vs {away} | {mk}: {name}{point_txt} @ {max_odds:.2f} | p~{p_med*100:.1f}%"
            picks.append(Pick(match_id=match_id, label=label, odds=max_odds, p_est=p_med, sport_key=sport_key))

    # Orden: “más probable dentro de cuotas altas”
    picks.sort(key=lambda p: (p.p_est, -p.odds), reverse=True)
    return picks

def build_acca(picks: List[Pick], target: float, legs_range: Tuple[int,int], tol: float = 0.18, beam: int = 250) -> List[Pick]:
    lo, hi = target*(1-tol), target*(1+tol)

    def score(p: Pick) -> float:
        return math.log(max(p.p_est, 1e-9))

    # Beam search
    states = [([], 0.0, 0.0, set())]  # (acca, log_odds, score_sum, used_match_ids)
    for p in picks:
        new_states = states[:]
        for acca, logo, sc, used in states:
            if p.match_id in used: 
                continue
            if len(acca) >= legs_range[1]:
                continue
            acc2 = acca + [p]
            log2 = logo + math.log(p.odds)
            sc2 = sc + score(p)
            used2 = set(used); used2.add(p.match_id)
            new_states.append((acc2, log2, sc2, used2))

        new_states.sort(key=lambda x: x[2], reverse=True)
        states = new_states[:beam]

    best = []
    best_sc = -1e18
    for acca, logo, sc, _ in states:
        if not (legs_range[0] <= len(acca) <= legs_range[1]): 
            continue
        odds_total = math.exp(logo)
        if lo <= odds_total <= hi and sc > best_sc:
            best = acca
            best_sc = sc
    return best

def main():
    all_picks: List[Pick] = []
    for sk in SPORT_KEYS:
        try:
            data = fetch_odds(sk)
            all_picks.extend(extract_candidates(data, sk))
        except Exception:
            continue

    if not all_picks:
        tg_send("No encontré candidatos hoy (o la API no devolvió datos).")
        return

    acca = build_acca(all_picks, TARGET_ODDS, LEGS_RANGE)
    if not acca:
        tg_send("No pude armar cuotón ~900 con los filtros actuales. (Prueba ampliando ligas o bajando filtros).")
        return

    odds_total = 1.0
    p_total = 1.0
    lines = []
    for i, p in enumerate(acca, 1):
        odds_total *= p.odds
        p_total *= p.p_est
        lines.append(f"{i}) {p.label}")

    msg = (
        f"🎯 CUOTÓN OBJETIVO ~{TARGET_ODDS:.0f}\n"
        f"Cuota aprox: {odds_total:.2f}\n"
        f"Prob. estimada (muy aproximada): {p_total*100:.4f}%\n\n"
        + "\n".join(lines)
        + "\n\nNota: p% es estimación por consenso de cuotas (MVP)."
    )
    tg_send(msg)

if __name__ == "__main__":
    main()
