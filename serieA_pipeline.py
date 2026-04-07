"""
Serie A Pipeline - Rating & Standings Prediction
=================================================
Tutto automatico — nessun CSV da scaricare a mano.

  - Rating squadre:     scaricato da understat.com (xG, gol, partite H/A)
  - Classifica:         scaricato da football-data.org (API v4)
  - Partite rimanenti:  scaricato da football-data.org (API v4)

Input richiesti (nella stessa cartella dello script):
  - Loghi_SerieA.csv
  - standings{N}.csv per ogni N in COMPARE_STAGES (giornate precedenti già salvate)

Modifica solo la sezione CONFIG qui sotto.
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import urllib.request
from PIL import Image
from scipy.stats import poisson
from highlight_text import fig_text, ax_text

BASE_DIR = Path(__file__).resolve().parent

def p(filename):
    return str(BASE_DIR / filename)


# ─────────────────────────────────────────────
# CONFIG — modifica qui
# ─────────────────────────────────────────────
API_KEY        = "6b8db78b7ba34a058417bdac8aec6a46"   # ← football-data.org
LOGHI_CSV      = "Loghi_SerieA.csv"
SEASON         = "2025"                  # anno di FINE stagione su Understat (2024/25 → 2025)

# Giornate precedenti già salvate da confrontare nel grafico
# La giornata corrente viene aggiunta automaticamente
COMPARE_STAGES = [17, 27, 28, 29, 30]
# ─────────────────────────────────────────────

API_BASE      = "https://api.football-data.org/v4"
COMPETITION   = "SA"
API_HEADERS   = {"X-Auth-Token": API_KEY}
UNDERSTAT_URL = f"https://understat.com/league/Serie_A/{SEASON}"

UNDERSTAT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Mappa nomi → nome interno univoco usato in tutto il pipeline
TEAM_NAME_MAP = {
    # Understat
    "Internazionale":           "Inter",
    "Milan":                    "Milan",
    "Hellas Verona":            "Verona",
    "Parma":                    "Parma",
    # football-data.org
    "FC Internazionale Milano": "Inter",
    "Inter Milan":              "Inter",
    "AC Milan":                 "Milan",
    "Hellas Verona FC":         "Verona",
    "Parma Calcio 1913":        "Parma",
    "US Lecce":                 "Lecce",
    "Cagliari Calcio":          "Cagliari",
    "Udinese Calcio":           "Udinese",
    "Genoa CFC":                "Genoa",
    "Bologna FC 1909":          "Bologna",
    "SS Lazio":                 "Lazio",
    "SSC Napoli":               "Napoli",
    "Juventus FC":              "Juventus",
    "AS Roma":                  "Roma",
    "Atalanta BC":              "Atalanta",
    "ACF Fiorentina":           "Fiorentina",
    "Torino FC":                "Torino",
    "US Sassuolo Calcio":       "Sassuolo",
    "Como 1907":                "Como",
    "US Cremonese":             "Cremonese",
    "Pisa SC":                  "Pisa",
    "AC Pisa 1909":             "Pisa",
    "Venezia FC":               "Venezia",
    "Empoli FC":                "Empoli",
    "AC Monza":                 "Monza",
    "Frosinone Calcio":         "Frosinone",
}

def normalize_team(name):
    return TEAM_NAME_MAP.get(name, name).strip()


# ── 0a. RATING DA UNDERSTAT ───────────────────────────────────────────────────

def fetch_ratings_from_understat():
    """
    Usa Playwright per aprire Understat in un browser headless,
    aspetta che il JS carichi i dati, poi estrae teamsData dal DOM.
    """
    print("⏳ Scarico statistiche da understat.com (browser headless)...")

    html = None
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = browser.new_page()
        # aspetta che la variabile teamsData sia disponibile nel JS
        page.goto(UNDERSTAT_URL, wait_until="networkidle", timeout=30000)
        # estrai teamsData direttamente dal contesto JS della pagina
        teams_data = page.evaluate("() => JSON.parse(JSON.stringify(teamsData))")
        browser.close()

    if not teams_data:
        raise RuntimeError("Playwright non ha trovato teamsData nella pagina.")

    rows = []
    for _, team_info in teams_data.items():
        raw_name = team_info["title"]
        name     = normalize_team(raw_name)
        history = team_info["history"]

        home_m = [m for m in history if m["h_a"] == "h"]
        away_m = [m for m in history if m["h_a"] == "a"]

        def agg(matches):
            return {
                "MP":  len(matches),
                "GF":  sum(int(m["scored"])  for m in matches),
                "GA":  sum(int(m["missed"])  for m in matches),
                "xG":  round(sum(float(m["xG"])  for m in matches), 2),
                "xGA": round(sum(float(m["xGA"]) for m in matches), 2),
            }

        h, a = agg(home_m), agg(away_m)
        rows.append({
            "team": name,
            "MPH": h["MP"], "MPA": a["MP"],
            "GFH": h["GF"], "GFA": a["GF"],
            "GAH": h["GA"], "GAA": a["GA"],
            "xGH": h["xG"], "xGA": a["xG"],
            "xGAH": h["xGA"], "xGAA": a["xGA"],
        })

    df = pd.DataFrame(rows)
    df.insert(3, "GF", df["GFH"] + df["GFA"])
    df.insert(4, "GA", df["GAH"] + df["GAA"])

    # offensive rating
    df = df.sort_values("GF", ascending=False).reset_index(drop=True)
    gf_mean = df["GF"].mean()
    df["off."] = [2.6 - (df.iloc[0, 3] - df.iloc[i, 3]) / gf_mean for i in range(len(df))]

    # defensive rating
    df = df.sort_values("GA", ascending=True).reset_index(drop=True)
    ga_mean = df["GA"].mean()
    df["def."] = [0.6 + (df.iloc[i, 4] - df.iloc[0, 4]) / ga_mean for i in range(len(df))]

    # SPI
    off_min = df["off."].min()
    def_max = df["def."].max()
    spi = []
    for i in range(len(df)):
        off_n = (df.iloc[i]["off."] - off_min) / (2.6 - off_min)
        def_n = (def_max - df.iloc[i]["def."]) / (def_max - 0.6)
        spi.append(57 + (83 - 57) * (0.5 * off_n + 0.5 * def_n))
    df["spi"] = spi

    rating = df[["team", "off.", "def.", "spi"]].copy()
    rating.to_csv(p("My_serieA_rating.csv"), index=False)
    print(f"✓ Rating calcolato per {len(rating)} squadre → My_serieA_rating.csv")
    return rating


# ── 0b. CLASSIFICA E CALENDARIO DA football-data.org ─────────────────────────

def fetch_standings():
    print("⏳ Scarico classifica da football-data.org...")
    resp = requests.get(
        f"{API_BASE}/competitions/{COMPETITION}/standings",
        headers=API_HEADERS, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()

    current_gw = data["season"]["currentMatchday"]
    rows = [
        {"team": normalize_team(e["team"]["name"]), "Points": float(e["points"])}
        for e in data["standings"][0]["table"]
    ]
    df = pd.DataFrame(rows)
    print(f"✓ Classifica scaricata — giornata corrente: {current_gw}")
    return df, current_gw


def fetch_remaining_fixtures(current_gw):
    print("⏳ Scarico partite rimanenti da football-data.org...")
    resp = requests.get(
        f"{API_BASE}/competitions/{COMPETITION}/matches",
        headers=API_HEADERS, params={"status": "SCHEDULED"}, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()

    rows = [
        {
            "HomeTeam": normalize_team(m["homeTeam"]["name"]),
            "AwayTeam": normalize_team(m["awayTeam"]["name"]),
        }
        for m in data["matches"] if m["matchday"] > current_gw
    ]
    df = pd.DataFrame(rows)
    print(f"✓ {len(df)} partite rimanenti scaricate (dalla giornata {current_gw + 1})")
    return df


# ── 1. PREVISIONE CLASSIFICA ──────────────────────────────────────────────────

def predict_standings(rating_df, standings_df, fixtures_df, output_gw):
    df    = rating_df.set_index("team")
    table = standings_df.copy()
    table["Points"] = table["Points"].astype(float)

    def predict(home, away):
        if home not in df.index or away not in df.index:
            print(f"  ⚠ Squadra non nel rating: '{home}' vs '{away}'")
            return (0.0, 0.0)
        lH = df.at[home, "off."] * df.at[away, "def."] * 1.1
        lA = df.at[away, "off."] * df.at[home, "def."] * 0.9
        if df.at[home, "spi"] <= df.at[away, "spi"]:
            d = (df.at[away, "spi"] - df.at[home, "spi"]) / 200
            lH *= (1 - d); lA *= (1 + d)
        else:
            d = (df.at[home, "spi"] - df.at[away, "spi"]) / 200
            lH *= (1 + d); lA *= (1 - d)
        pH = pA = pT = 0.0
        for x in range(11):
            for y in range(11):
                prob = poisson.pmf(x, lH) * poisson.pmf(y, lA)
                if x == y:   pT += prob
                elif x > y:  pH += prob
                else:        pA += prob
        return (3 * pH + pT, 3 * pA + pT)

    for _, row in fixtures_df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        sH, sA = predict(h, a)
        table.loc[table.team == h, "Points"] += sH
        table.loc[table.team == a, "Points"] += sA

    table = table.sort_values("Points", ascending=True).reset_index(drop=True)
    table.index = np.arange(1, len(table) + 1)
    out_name = f"standings{output_gw}.csv"
    table.round(2).to_csv(p(out_name), index=False)
    print(f"✓ Classifica prevista salvata → {out_name}")
    return table


# ── 2. GRAFICO COMPARATIVO ────────────────────────────────────────────────────

def plot_standings(loghi_csv, compare_stages, output_image):
    frames = {}
    for gw in compare_stages:
        tmp     = pd.read_csv(p(f"standings{gw}.csv"))
        pts_col = [c for c in tmp.columns if c.lower().startswith('point')][0]
        tmp     = tmp[['team', pts_col]].rename(columns={pts_col: f'Points{gw}'})
        tmp['team'] = tmp['team'].apply(normalize_team)   # ← normalizza subito
        frames[gw] = tmp

    df = frames[compare_stages[0]]
    for gw in compare_stages[1:]:
        df = df.merge(frames[gw], on='team')


    dfl = pd.read_csv(p(loghi_csv))

    # debug: mostra squadre che non trovano corrispondenza nei loghi
    missing = set(df['team']) - set(dfl['team'])
    if missing:
        print(f"  ⚠ Squadre senza corrispondenza in {loghi_csv}: {missing}")
        print(f"  Nomi disponibili in {loghi_csv}: {sorted(dfl['team'].tolist())}")

    df = df.merge(dfl, on='team')

    pts_cols = [f'Points{g}' for g in compare_stages]
    df = df.sort_values(pts_cols[-1], ascending=True).reset_index(drop=True)

    # layout
    n_stages   = len(compare_stages)
    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    nrows      = 20
    LOGO_X     = 0.4
    NAME_X     = 1
    COL_START  = 1.8
    COL_STEP   = 0.8
    DELTA_OFF  = 0.3

    ncols = COL_START + n_stages * COL_STEP + 0.5
    x_pts = [COL_START + i * COL_STEP for i in range(n_stages)]

    fig = plt.figure(figsize=(16, 22), dpi=200)
    ax  = plt.subplot(111, facecolor="#EFE9E6")
    ax.set_xlim(0, ncols)
    ax.set_ylim(-0.5, nrows + 2.5)

    DC_to_FC  = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    DC_to_NFC = lambda xy: FC_to_NFC(DC_to_FC(xy))

    for y in range(nrows):
        row_y = y + 0.5

        # logo
        team_id   = df['team_id'].iloc[y]
        ax_coords = DC_to_NFC([LOGO_X, y + 0.05])
        logo_ax   = fig.add_axes(
            [ax_coords[0] - 0.018, ax_coords[1], 0.036, 0.036], anchor="C"
        )
        try:
            url  = f"{fotmob_url}{int(team_id)}.png"
            icon = Image.open(urllib.request.urlopen(url))
            logo_ax.imshow(icon)
        except Exception as e:
            print(f"  Logo non caricato per team_id={team_id}: {e}")
        logo_ax.axis("off")

        # nome squadra
        ax.annotate(xy=(NAME_X, row_y), text=df['team'].iloc[y],
                    ha='center', va='center', size=13, weight='bold')

        # punti + delta
        for i, col in enumerate(pts_cols):
            x   = x_pts[i]
            val = df[col].iloc[y]
            txt = ax.annotate(xy=(x, row_y), text=f'{val}',
                              ha='center', va='center', size=13, weight='bold')
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground="white"),
                path_effects.Normal()
            ])
            if i > 0:
                prev  = df[pts_cols[i-1]].iloc[y]
                diff  = round(val - prev, 2)
                color = 'green' if diff >= 0 else 'red'
                sign  = '+' if diff >= 0 else ''
                ax_text(
                    x=x - DELTA_OFF, y=row_y,
                    s=f'<{sign}{diff}>',
                    ha='center', va='center',
                    highlight_textprops=[{'size': 8, 'color': color}],
                    size=8, ax=ax
                )

    # fasce colorate
    zones = [
        (0,  3,  '#ca7161'),
        (3,  4,  '#eab7ad'),
        (13, 14, '#d2dad5'),
        (14, 16, '#a6b6ad'),
        (16, 19, '#537162'),
        (19, 20, '#003822'),
    ]
    for y1, y2, color in zones:
        ax.fill_between([0, ncols], y1, y2,
                        color=color, alpha=0.5, ec='None', zorder=2)

    ax.plot([0, ncols], [nrows, nrows], lw=1.5, color='black', zorder=4)
    ax.plot([0, ncols], [0, 0],         lw=1.5, color='black', zorder=4)
    for row in range(1, nrows):
        ax.plot([0, ncols], [row, row], lw=1.15, color='gray', ls=':', zorder=3)

    ax.annotate(xy=(NAME_X, nrows + 0.8), text="TEAM",
                weight="bold", ha="center", size=12)
    for i, gw in enumerate(compare_stages):
        ax.annotate(xy=(x_pts[i], nrows + 0.8),
                    text=f"Points\nafter\n{gw} stage",
                    weight="bold", ha="center", size=11)

    ax.set_axis_off()
    fig_text(x = 0.15, y = 0.87, 
        s = "Serie A Standing Prediction after multiple stages",
        va = "bottom", ha = "left", fontsize = 20, color = "black", weight = "bold")   

    plt.savefig(p(output_image), bbox_inches='tight', dpi=200)
    print(f"✓ Grafico salvato → {output_image}")
    #plt.show()

# ── 3. GRAFICO ANDAMENTO STAGIONALE ──────────────────────────────────────────

def find_all_stages():
    """
    Scansiona BASE_DIR e ritorna la lista ordinata di tutte le giornate
    per cui esiste un file standings{N}.csv.
    """
    found = []
    for f in BASE_DIR.glob("standings*.csv"):
        m = re.match(r"standings(\d+)\.csv", f.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(found)




# Palette colori per le 20 squadre (dal peggiore al migliore)
CHART_COLORS = [
    '#52170b','#7f2b19','#b03c23','#c0783e','#b57b38','#a97d35',
    '#9e7f34','#928134','#878137','#7c823a','#71823f','#668244',
    '#5c814a','#528050','#497f56','#407d5b','#387b61','#246963',
    '#184642','#0C2321',
]


def _load_chart_data(loghi_csv, all_stages):
    """
    Carica tutti gli standings{N}.csv, li unisce e aggiunge colori e team_id.
    Ritorna un DataFrame con colonne: team, <gw1>, <gw2>, ..., color, team_id
    Le colonne GW sono nominate con il numero intero della giornata.
    """
    import matplotlib.ticker as ticker

    # carica e merge
    frames = {}
    for gw in all_stages:
        tmp     = pd.read_csv(p(f"standings{gw}.csv"))
        pts_col = [c for c in tmp.columns if c.lower().startswith("point")][0]
        tmp     = tmp[["team", pts_col]].rename(columns={pts_col: str(gw)})
        tmp["team"] = tmp["team"].apply(normalize_team)
        frames[gw] = tmp

    df = frames[all_stages[0]]
    for gw in all_stages[1:]:
        df = df.merge(frames[gw], on="team")

    # merge loghi
    dfl = pd.read_csv(p(loghi_csv))
    df  = df.merge(dfl, on="team")

    # ordina per ultima giornata e assegna colori
    last_col = str(all_stages[-1])
    df = df.sort_values(by=last_col, ascending=True).reset_index(drop=True)
    df.insert(len(df.columns) - 1, "color", CHART_COLORS[:len(df)])

    return df


def _add_logo(fig, ax, team_id, x_data, y_data, offset=(-0.020, -0.020), grey=False):
    fotmob_url = "https://images.fotmob.com/image_resources/logo/teamlogo/"
    DC_to_FC   = ax.transData.transform
    FC_to_NFC  = fig.transFigure.inverted().transform
    coords     = FC_to_NFC(DC_to_FC([x_data, y_data]))
    logo_ax    = fig.add_axes(
        [coords[0] + offset[0], coords[1] + offset[1], 0.04, 0.04],
        anchor="C", zorder=6
    )
    try:
        icon = Image.open(urllib.request.urlopen(f"{fotmob_url}{int(team_id)}.png"))
        if grey:
            icon = icon.convert("LA")
        logo_ax.imshow(icon)
    except Exception as e:
        print(f"  Logo non caricato per team_id={team_id}: {e}")
    logo_ax.axis("off")


def plot_season_chart(loghi_csv, all_stages, output_image):
    """
    Grafico andamento stagionale: tutte le squadre, linee colorate.
    Asse X = giornate, asse Y = punti previsti a fine stagione.
    """
    import matplotlib.ticker as ticker

    df     = _load_chart_data(loghi_csv, all_stages)
    stages = [str(g) for g in all_stages]
    x_vals = [int(g) for g in all_stages]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=400)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlim(x_vals[0] - 1, x_vals[-1] + 1)
    ax.set_ylim(15, 95)
    ax.grid(ls="--", color="#efe9e6", zorder=2)

    # linea zona retrocessione/Champions
    ax.plot([x_vals[0], x_vals[-1]], [35, 35], color="black", ls="dashed", lw=1)
    ax.fill_between([x_vals[0]-1, x_vals[-1]+1], 35, 100,
                    color="#336699", alpha=0.04, ec="None", hatch=".....", zorder=1)
    ax.fill_between([x_vals[0]-1, x_vals[-1]+1], 35, 15,
                    color="#DA4167", alpha=0.04, ec="None", hatch=".....", zorder=1)

    for i in range(len(df) - 1, -1, -1):
        row     = df.iloc[i]
        y_vals  = [row[c] for c in stages]
        color   = row["color"]
        team_id = row["team_id"]

        ax.plot(x_vals, y_vals, lw=1.5, color=color, zorder=5,
                markevery=[-1], marker="o", ms=4, mfc="white")

        # logo sul primo punto
        _add_logo(fig, ax, team_id, x_vals[-1] + 0.5, y_vals[-1],
                  offset=(-0.020, -0.020))

    fig_text(x=0.12, y=0.96, s="Serie A — Andamento Classifica Prevista",
             va="bottom", ha="left", fontsize=14, color="black", weight="bold")

    plt.savefig(p(output_image), bbox_inches="tight", dpi=200)
    print(f"✓ Grafico stagionale salvato → {output_image}")
    #plt.show()
    plt.close()


def plot_team_chart(loghi_csv, all_stages, team_name, output_image):
    """
    Grafico focus su una squadra: evidenziata a colori, le altre in grigio.
    Mostra banda min/max, media tratteggiata, etichette punti.
    """
    import matplotlib.ticker as ticker

    df     = _load_chart_data(loghi_csv, all_stages)
    stages = [str(g) for g in all_stages]
    x_vals = [int(g) for g in all_stages]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=400)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlim(x_vals[0] - 0.5, x_vals[-1] + 1)
    ax.set_ylim(25, 100)
    ax.grid(ls="--", color="#efe9e6", zorder=2)

    for _, row in df.iterrows():
        team    = row["team"]
        team_id = row["team_id"]
        y_vals  = [row[c] for c in stages]

        if team == team_name:
            color = row["color"]
            ax.plot(x_vals, y_vals, lw=1.5, color=color, zorder=5,
                    markevery=[-1], marker="o", ms=4, mfc="white")
            ax.fill_between(x_vals, min(y_vals), max(y_vals),
                            color=color, alpha=0.1, ec="None", hatch=".....", zorder=1)
            ax.plot([x_vals[0], x_vals[-1]], [min(y_vals), min(y_vals)],
                    color=color, ls="dashed", lw=1)
            ax.plot([x_vals[0], x_vals[-1]], [max(y_vals), max(y_vals)],
                    color=color, ls="dashed", lw=1)
            # etichette punti
            for xi, yi in zip(x_vals, y_vals):
                ax.text(xi - 0.25, yi + 1, round(yi, 1), fontsize=6, zorder=7)
            # logo
            _add_logo(fig, ax, team_id, x_vals[0], y_vals[0],
                      offset=(+0.015, -0.020))
            # media
            mean_val = round(sum(y_vals) / len(y_vals), 1)
            ax.plot([x_vals[0], x_vals[-1]], [mean_val, mean_val],
                    color="grey", ls="dashed", lw=1)
        else:
            ax.plot(x_vals, y_vals, lw=1.5, color="grey", alpha=0.25)
            _add_logo(fig, ax, team_id, x_vals[0], y_vals[0],
                      offset=(+0.015, -0.020), grey=True)

    ax.fill_between([x_vals[0]-0.5, x_vals[-1]+1], 35, 100,
                    color="#336699", alpha=0.05, ec="None", hatch=".....", zorder=1)
    ax.fill_between([x_vals[0]-0.5, x_vals[-1]+1], 35, 15,
                    color="#DA4167", alpha=0.05, ec="None", hatch=".....", zorder=1)

    fig_text(x=0.12, y=0.96, s=f"Serie A — Focus: {team_name}",
             va="bottom", ha="left", fontsize=14, color="black", weight="bold")

    plt.savefig(p(output_image), bbox_inches="tight", dpi=200)
    print(f"✓ Grafico focus {team_name} salvato → {output_image}")
    #plt.show()
    plt.close()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Rating da Understat
    rating_df = fetch_ratings_from_understat()

    # 2. Classifica e partite rimanenti da football-data.org
    standings_df, current_gw = fetch_standings()
    fixtures_df = fetch_remaining_fixtures(current_gw)

    # 3. Previsione classifica finale
    predict_standings(rating_df, standings_df, fixtures_df, current_gw)

    # 4. Grafico comparativo (aggiunge giornata corrente se non già presente)
    stages = COMPARE_STAGES.copy()
    if current_gw not in stages:
        stages.append(current_gw)

    plot_standings(LOGHI_CSV, stages, f"standings_prediction_{current_gw}.png")

    # 5 & 6. Grafici stagionali: usa TUTTI i standings{N}.csv trovati in cartella
    all_stages = find_all_stages()
    print(f"✓ Giornate trovate per grafici stagionali: {all_stages}")

    plot_season_chart(LOGHI_CSV, all_stages, f"season_chart_{current_gw}.png")

    # Per il grafico focus, decommentare e cambiare il nome squadra:
    # plot_team_chart(LOGHI_CSV, all_stages, "Napoli", f"team_chart_Napoli_{current_gw}.png")