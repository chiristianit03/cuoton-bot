import os, math, requests, datetime
from dataclasses import dataclass
from typing import List, Tuple

ODDS_API_KEY = os.environ["ODDS_API_KEY"]
TG_BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]
GITHUB_SHA = os.environ.get("GITHUB_SHA", "")[:7]

REGIONS = "uk"          # recomendado para soccer en v4
MARKETS = "h2h"         # mercado featured más estable en v4
ODDS_FORMAT = "decimal"

# Empieza con pocas ligas para no gastar créditos
SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
]

# Para cuotón estilo 900 necesitas cuotas por pata altas.
# Con h2h, normalmente: underdogs/draw caen en 3.0–8.0
ODDS_MIN = 3.0
ODDS_MAX = 12.0

TARGET_ODDS = 900.0
LEGS_RANGE = (4, 10)     # con h2h a veces necesitas más patas para llegar a 900
ODDS_TOL = 0.20          # ±20% alrededor del objetivo
BEAM_WIDTH = 250

@dataclass(frozen=True)
class Pick:
    match_id: str
    label: str
    odds: float
    p_est: float   # estimación por consenso (implied prob)

def tg_send(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg}, timeout=25)

def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 0.0

def fetch_odds(sport_key: str) -> list:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=25)
    used = r.headers.get("x-requests-used")
    rem = r.headers.get("x-requests-remaining")

    if r.status_code != 200:
        tg_send(f"❌ API ERROR {sport_key} HTTP {r.status_code} used={used} rem={rem}\n{r.text[:350]}")
        return []

    data = r.json()
    bm0 = len(data[0].get("bookmakers", [])) if data else 0
    tg_send(f"✅ {sport_key}: events={len(data)} first_bookmakers={bm0} used={used} rem={rem} sha={GITHUB_SHA}")
    return data

def extract_candidates(data: list) -> List[Pick]:
    picks: List[Pick] = []
    events_with_bm = 0
    total_outcomes = 0

    for ev in data:
        match_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not match_id or not home or not away:
            continue

        bookmakers = ev.get("bookmakers", [])
        if not bookmakers:
            continue

        events_with_bm += 1

        # juntamos odds de cada outcome (home/away/draw) de distintos bookmakers
        # key: outcome_name -> list of odds
        odds_map = {}
        for bm in bookmakers:
            for m in bm.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name = o.get("name")
                    price = o.get("price")
                    if not name or not isinstance(price, (int, float)):
                        continue
                    odds_map.setdefault(name, []).append(float(price))

        for name, odds_list in odds_map.items():
            if not odds_list:
                continue
            total_outcomes += 1
            best_odds = max(odds_list)
            if not (ODDS_MIN <= best_odds <= ODDS_MAX):
                continue

            # estimación simple: usamos implied prob con la "mejor cuota" (optimista)
            # (en MVP solo sirve para ranking; luego metemos modelo real)
            p = implied_prob(best_odds)

            label = f"{home} vs {away} | H2H: {name} @ {best_odds:.2f} (p~{p*100:.1f}%)"
            picks.append(Pick(match_id=match_id, label=label, odds=best_odds, p_est=p))

    tg_send(f"📊 extract: events_with_bookmakers={events_with_bm} outcomes_seen={total_outcomes} picks_after_filter={len(picks)}")
    picks.sort(key=lambda x: (x.p_est, -x.odds), reverse=True)
    return picks

def build_acca(picks: List[Pick]) -> List[Pick]:
    lo, hi = TARGET_ODDS*(1-ODDS_TOL), TARGET_ODDS*(1+ODDS_TOL)

    def score(p: Pick) -> float:
        return math.log(max(p.p_est, 1e-12))

    states = [([], 0.0, 0.0, set())]  # (acca, log_odds, score_sum, used_match_ids)
    for p in picks:
        new_states = states[:]
        for acca, logod, sc, used in states:
            if p.match_id in used: 
                continue
            if len(acca) >= LEGS_RANGE[1]:
                continue

            acc2 = acca + [p]
            log2 = logod + math.log(p.odds)
            sc2 = sc + score(p)
            used2 = set(used); used2.add(p.match_id)
            new_states.append((acc2, log2, sc2, used2))

        new_states.sort(key=lambda x: x[2], reverse=True)
        states = new_states[:BEAM_WIDTH]

    best = []
    best_sc = -1e18
    for acca, logod, sc, _ in states:
        if not (LEGS_RANGE[0] <= len(acca) <= LEGS_RANGE[1]):
            continue
        odds_total = math.exp(logod)
        if lo <= odds_total <= hi and sc > best_sc:
            best, best_sc = acca, sc
    return best

def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tg_send(f"🚀 RUN {now} sha={GITHUB_SHA} regions={REGIONS} markets={MARKETS}")

    all_picks: List[Pick] = []
    for sk in SPORT_KEYS:
        data = fetch_odds(sk)
        all_picks.extend(extract_candidates(data))

    if not all_picks:
        tg_send("⚠️ No encontré picks tras filtros. (Baja ODDS_MIN a 2.5 o añade más ligas)")
        return

    # Manda top 10 candidatos para verificar que ya está funcionando
    top = all_picks[:10]
    msg = "✅ TOP CANDIDATOS (verificación)\n" + "\n".join([f"{i+1}) {p.label}" for i, p in enumerate(top)])
    tg_send(msg)

    acca = build_acca(all_picks)
    if not acca:
        tg_send("⚠️ No pude armar cuotón ~900 con lo disponible. (Sube LEGS_RANGE o baja ODDS_MIN)")
        return

    odds_total = 1.0
    p_total = 1.0
    lines = []
    for i, p in enumerate(acca, 1):
        odds_total *= p.odds
        p_total *= p.p_est
        lines.append(f"{i}) {p.label}")

    tg_send(
        f"🎯 CUOTÓN ~{TARGET_ODDS:.0f}\nCuota: {odds_total:.2f}\n"
        f"Prob aprox (solo implied): {p_total*100:.4f}%\n\n" + "\n".join(lines)
    )

if __name__ == "__main__":
    main()

