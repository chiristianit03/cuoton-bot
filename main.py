import os, requests, datetime

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
GITHUB_SHA = os.environ.get("GITHUB_SHA", "")[:7]

def tg_send(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg}, timeout=20)

def get(url: str, params: dict):
    r = requests.get(url, params=params, timeout=25)
    return r.status_code, r.headers, r.text[:600]

def main():
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tg_send(f"🧪 DIAGNÓSTICO INICIO | {now} | sha={GITHUB_SHA}")

    if not ODDS_API_KEY:
        tg_send("❌ Falta ODDS_API_KEY (secrets).")
        return

    # 1) Endpoint gratis (no consume credits)
    url_sports = "https://api.the-odds-api.com/v4/sports/"
    code, headers, body = get(url_sports, {"apiKey": ODDS_API_KEY})
    tg_send(f"1) /sports HTTP={code}\nbody={body[:250]}")

    # 2) EPL odds (sí consume credits)
    url_epl = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    code2, headers2, body2 = get(url_epl, params)
    used = headers2.get("x-requests-used")
    rem = headers2.get("x-requests-remaining")
    tg_send(f"2) EPL odds HTTP={code2} used={used} remaining={rem}\nbody={body2[:250]}")

if __name__ == "__main__":
    main()
