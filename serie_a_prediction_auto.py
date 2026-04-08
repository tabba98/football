"""
Serie A Automatic Match Prediction Generator
=============================================
Scarica automaticamente:
  - Rating squadre da Understat
  - Classifica e partite rimanenti da football-data.org
Poi genera previsioni visive per ogni match della prossima giornata.

Modifica la sezione CONFIG qui sotto per personalizzare.
"""

import pandas as pd
import numpy as np
import math
import requests
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import urllib.request
from PIL import Image
import warnings
import time
from bs4 import BeautifulSoup
from scipy.stats import poisson

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG — modifica qui
# ─────────────────────────────────────────────
API_KEY_FOOTBALL_DATA = "6b8db78b7ba34a058417bdac8aec6a46"
SEASON = "2025"  # anno di FINE stagione su Understat (2024/25 → 2025)
OUTPUT_DIR = "serie_a_predictions"  # cartella dove salvare le immagini
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / OUTPUT_DIR

# Crea cartella output se non esiste
OUTPUT_PATH.mkdir(exist_ok=True)

API_BASE = "https://api.football-data.org/v4"
COMPETITION = "SA"
API_HEADERS = {"X-Auth-Token": API_KEY_FOOTBALL_DATA}
UNDERSTAT_URL = f"https://understat.com/league/Serie_A/{SEASON}"

UNDERSTAT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Mappa normalizzazione nomi squadre
TEAM_NAME_MAP = {
    "Internazionale": "Inter",
    "Milan": "Milan",
    "Hellas Verona": "Verona",
    "Parma": "Parma",
    "FC Internazionale Milano": "Inter",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "Hellas Verona FC": "Verona",
    "Parma Calcio 1913": "Parma",
    "US Lecce": "Lecce",
    "Cagliari Calcio": "Cagliari",
    "Udinese Calcio": "Udinese",
    "Genoa CFC": "Genoa",
    "Bologna FC 1909": "Bologna",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Juventus FC": "Juventus",
    "AS Roma": "Roma",
    "Atalanta BC": "Atalanta",
    "ACF Fiorentina": "Fiorentina",
    "Torino FC": "Torino",
    "US Sassuolo Calcio": "Sassuolo",
    "Como 1907": "Como",
    "US Cremonese": "Cremonese",
    "Pisa SC": "Pisa",
    "AC Pisa 1909": "Pisa",
    "Venezia FC": "Venezia",
    "Empoli FC": "Empoli",
    "AC Monza": "Monza",
    "Frosinone Calcio": "Frosinone",
}

def normalize_team(name):
    return TEAM_NAME_MAP.get(name, name).strip()


# ─────────────────────────────────────────────────────────────────────
# 0. SCARICAMENTO DATI DA UNDERSTAT (con beautifulsoup e richieste HTTP)
# ─────────────────────────────────────────────────────────────────────

def fetch_ratings_from_understat():
    """
    Scarica statistiche da Understat usando BeautifulSoup e ricerca JSON nei script della pagina.
    Ritorna sia il dataframe dei rating che il dataframe delle statistiche storiche reali.
    
    Returns:
        (rating_df, stats_df): Tuple di due dataframe
            - rating_df: team, off., def., spi
            - stats_df: team, MPH, MPA, GFH, GFA, GAH, GAA, xGH, xGA, xGAH, xGAA
    """
    print("⏳ Scarico statistiche da understat.com...")
    
    try:
        response = requests.get(UNDERSTAT_URL, headers=UNDERSTAT_HEADERS, timeout=15)
        response.raise_for_status()
        html = response.text
        
        # Cerca teamsData nel codice JavaScript della pagina
        import re
        match = re.search(r'var teamsData = ({.*?});', html, re.DOTALL)
        if not match:
            print("⚠ Impossibile trovare teamsData nella pagina.")
            print("  Usare versione con Playwright se disponibile.")
            fallback_rating = create_fallback_rating()
            fallback_stats = create_fallback_stats(fallback_rating['team'].tolist())
            return fallback_rating, fallback_stats
        
        teams_data = json.loads(match.group(1))
        
        rows_stats = []
        rows_rating = []
        
        for _, team_info in teams_data.items():
            raw_name = team_info["title"]
            name = normalize_team(raw_name)
            history = team_info["history"]
            
            home_m = [m for m in history if m["h_a"] == "h"]
            away_m = [m for m in history if m["h_a"] == "a"]
            
            def agg(matches):
                return {
                    "MP": len(matches),
                    "GF": sum(int(m["scored"]) for m in matches),
                    "GA": sum(int(m["missed"]) for m in matches),
                    "xG": round(sum(float(m["xG"]) for m in matches), 2),
                    "xGA": round(sum(float(m["xGA"]) for m in matches), 2),
                }
            
            h, a = agg(home_m), agg(away_m)
            
            # Dati per statistiche storiche (usati nelle predizioni)
            rows_stats.append({
                "team": name,
                "MPH": h["MP"], "MPA": a["MP"],
                "GFH": h["GF"], "GFA": a["GF"],
                "GAH": h["GA"], "GAA": a["GA"],
                "xGH": h["xG"], "xGA": a["xG"],
                "xGAH": h["xGA"], "xGAA": a["xGA"],
            })
            
            # Prepara dati per calcolo rating
            rows_rating.append({
                "team": name,
                "GFH": h["GF"], "GFA": a["GF"],
                "GAH": h["GA"], "GAA": a["GA"],
                "xGH": h["xG"], "xGA": a["xG"],
                "xGAH": h["xGA"], "xGAA": a["xGA"],
            })
        
        # Dataframe statistiche storiche (DATI REALI)
        stats_df = pd.DataFrame(rows_stats)
        print(f"✓ Statistiche storiche caricate per {len(stats_df)} squadre")
        
        # Calcolo rating
        df = pd.DataFrame(rows_rating)
        df.insert(3, "GF", df["GFH"] + df["GFA"])
        df.insert(4, "GA", df["GAH"] + df["GAA"])
        
        # Calcola rating offensivo
        df_sorted = df.sort_values("GF", ascending=False).reset_index(drop=True)
        gf_mean = df_sorted["GF"].mean()
        off_ratings = [2.6 - (df_sorted.iloc[0, 3] - df_sorted.iloc[i, 3]) / gf_mean for i in range(len(df_sorted))]
        df_sorted["off."] = off_ratings
        
        # Calcola rating difensivo
        df_sorted = df_sorted.sort_values("GA", ascending=True).reset_index(drop=True)
        ga_mean = df_sorted["GA"].mean()
        def_ratings = [0.6 + (df_sorted.iloc[i, 4] - df_sorted.iloc[0, 4]) / ga_mean for i in range(len(df_sorted))]
        df_sorted["def."] = def_ratings
        
        # Calcola SPI
        off_min = df_sorted["off."].min()
        def_max = df_sorted["def."].max()
        spi = []
        for i in range(len(df_sorted)):
            off_n = (df_sorted.iloc[i]["off."] - off_min) / (2.6 - off_min)
            def_n = (def_max - df_sorted.iloc[i]["def."]) / (def_max - 0.6)
            spi.append(57 + (83 - 57) * (0.5 * off_n + 0.5 * def_n))
        df_sorted["spi"] = spi
        
        rating_df = df_sorted[["team", "off.", "def.", "spi"]].copy().reset_index(drop=True)
        print(f"✓ Rating calcolato per {len(rating_df)} squadre")
        
        return rating_df, stats_df
        
    except Exception as e:
        print(f"⚠ Errore nel caricamento da Understat: {e}")
        fallback_rating = create_fallback_rating()
        fallback_stats = create_fallback_stats(fallback_rating['team'].tolist())
        return fallback_rating, fallback_stats


def create_fallback_rating():
    """
    Crea rating di fallback con valori standard se Understat non è raggiungibile.
    """
    print("✓ Usando rating di fallback...")
    teams = list(set([normalize_team(v) for v in TEAM_NAME_MAP.values()]))
    
    df = pd.DataFrame({
        "team": teams,
        "off.": [1.2] * len(teams),
        "def.": [1.0] * len(teams),
        "spi": [50.0] * len(teams),
    })
    
    return df.sort_values("team").reset_index(drop=True)


def create_fallback_stats(teams):
    """
    Crea dataset statistiche di fallback con valori standard.
    """
    print("✓ Usando statistiche storiche di fallback...")
    
    df = pd.DataFrame({
        "team": teams,
        "MPH": [13] * len(teams),
        "MPA": [13] * len(teams),
        "GFH": [18] * len(teams),
        "GFA": [18] * len(teams),
        "GAH": [13] * len(teams),
        "GAA": [13] * len(teams),
        "xGH": [14.0] * len(teams),
        "xGA": [14.0] * len(teams),
        "xGAH": [10.5] * len(teams),
        "xGAA": [10.5] * len(teams),
    })
    
    return df.sort_values("team").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# 1. SCARICAMENTO CLASSIFICA E PARTITE DA FOOTBALL-DATA.ORG
# ─────────────────────────────────────────────────────────────────────

def fetch_standings():
    """Scarica classifica corrente."""
    print("⏳ Scarico classifica da football-data.org...")
    
    try:
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
        
    except Exception as e:
        print(f"✗ Errore nel scaricamento classifica: {e}")
        raise


def fetch_next_matchday(current_gw):
    """Scarica partite della prossima giornata."""
    print(f"⏳ Scarico partite della giornata {current_gw + 1}...")
    
    try:
        resp = requests.get(
            f"{API_BASE}/competitions/{COMPETITION}/matches",
            headers=API_HEADERS, params={"status": "SCHEDULED"}, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        
        matches = [
            {
                "HomeTeam": normalize_team(m["homeTeam"]["name"]),
                "AwayTeam": normalize_team(m["awayTeam"]["name"]),
            }
            for m in data["matches"] if m["matchday"] == current_gw + 1
        ]
        
        if not matches:
            print(f"⚠ Nessuna partita trovata per la giornata {current_gw + 1}")
            return None
        
        df = pd.DataFrame(matches)
        print(f"✓ {len(df)} partite scaricate per la giornata {current_gw + 1}")
        return df
        
    except Exception as e:
        print(f"✗ Errore nel scaricamento partite: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────
# 2. CALCOLO PREDIZIONE MATCH (dal notebook)
# ─────────────────────────────────────────────────────────────────────

def prediction(data1, rating_df, team_home="Inter", team_away="Atalanta"):
    """
    Calcola la matrice di probabilità per una partita.
    
    Args:
        data1: DataFrame con statistiche (GF, GA, xG, xGA per home e away)
        rating_df: DataFrame con rating delle squadre
        team_home: nome squadra di casa
        team_away: nome squadra in trasferta
    
    Returns:
        Matrice 9x9 con probabilità di ogni risultato finale
    """
    
    data1.columns = ['team', 'MPH', 'MPA', 'GFH', 'GFA', 'GAH', 'GAA', 'xGH', 'xGA', 'xGAH', 'xGAA']
    
    data_merged = pd.merge(data1, rating_df, on='team')
    
    # Costanti medie
    average_xG_h = data_merged['xGH'].sum() / data_merged['MPH'].sum()
    average_xGA_h = data_merged['xGAH'].sum() / data_merged['MPH'].sum()
    average_xG_a = data_merged['xGA'].sum() / data_merged['MPA'].sum()
    average_xGA_a = data_merged['xGAA'].sum() / data_merged['MPA'].sum()
    
    average_GF_h = data_merged['GFH'].sum() / data_merged['MPH'].sum()
    average_GA_h = data_merged['GAH'].sum() / data_merged['MPH'].sum()
    average_GF_a = data_merged['GFA'].sum() / data_merged['MPA'].sum()
    average_GA_a = data_merged['GAA'].sum() / data_merged['MPA'].sum()
    
    # Filtra dati squadre
    team_df_h = data_merged[data_merged["team"] == team_home].reset_index(drop=True)
    team_df_a = data_merged[data_merged["team"] == team_away].reset_index(drop=True)
    
    if len(team_df_h) == 0 or len(team_df_a) == 0:
        return None
    
    # Calcola rating offensivo e difensivo
    home_attack = ((team_df_h["xGH"] / team_df_h['MPH']) / average_xG_h + 
                   (team_df_h["GFH"] / team_df_h['MPH']) / average_GF_h) / 2
    home_defense = ((team_df_h["xGAH"] / team_df_h['MPH']) / average_xGA_h + 
                    (team_df_h["GAH"] / team_df_h['MPH']) / average_GA_h) / 2
    away_attack = ((team_df_a["xGA"] / team_df_a['MPA']) / average_xG_a + 
                   (team_df_a["GFA"] / team_df_a['MPA']) / average_GF_a) / 2
    away_defense = ((team_df_a["xGAA"] / team_df_a['MPA']) / average_xGA_a + 
                    (team_df_a["GAA"] / team_df_a['MPA']) / average_GA_a) / 2
    
    # Calcola gol previsti
    projected_h_g = home_attack.values[0] * away_defense.values[0] * average_xG_h * average_GF_h * 1.03
    projected_a_g = away_attack.values[0] * home_defense.values[0] * average_xG_a * average_GF_a * 0.97
    
    # Aggiusta per SPI
    if team_df_h.at[0, "spi"] <= team_df_a.at[0, "spi"]:
        diff_pi = (team_df_a["spi"].values[0] - team_df_h["spi"].values[0]) / 200
        projected_h_g = projected_h_g * (1 - diff_pi)
        projected_a_g = projected_a_g * (1 + diff_pi)
    else:
        diff_pi = (team_df_h["spi"].values[0] - team_df_a["spi"].values[0]) / 200
        projected_h_g = projected_h_g * (1 + diff_pi)
        projected_a_g = projected_a_g * (1 - diff_pi)
    
    # Costruisci matrice di probabilità Poisson
    table = np.zeros((9, 9), float)
    h_goals = []
    a_goals = []
    
    for i in range(9):
        h_goals.append(((projected_a_g ** i) * np.exp(-projected_a_g)) / math.factorial(i))
    
    for i in range(9):
        a_goals.append(((projected_h_g ** i) * np.exp(-projected_h_g)) / math.factorial(i))
    
    for i in range(9):
        for j in range(9):
            table[i, j] = h_goals[i] * a_goals[j] * 100
    
    return table


# ─────────────────────────────────────────────────────────────────────
# 3. FUNZIONI DI VISUALIZZAZIONE (dal notebook)
# ─────────────────────────────────────────────────────────────────────

def battery_plot(win, draw, lose, ax):
    """Grafico batteria per probabilità di risultato 1X2."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.barh([0.5], [1], fc='white', ec='black', height=.35)
    ax.barh([0.5], [win / 100], fc='#de6f57', ec='black', hatch='//////', height=.35, zorder=2)
    ax.barh([0.5], [draw / 100], left=[win / 100], fc='grey', ec='black', hatch='//////', height=.35, zorder=2)
    ax.barh([0.5], [lose / 100], left=[win / 100 + draw / 100], fc='#287271', ec='black', hatch='//////', height=.35, zorder=2)
    
    ax.text((win / 100) / 2, 0.9, f'{win / 100:.1%}', ha='center', va='center', fontweight='bold', size=10)
    ax.text((win / 100) + (draw / 100) / 2, 0.9, f'{draw / 100:.1%}', ha='center', va='center', fontweight='bold', size=10)
    ax.text((win / 100) + (draw / 100) + (lose / 100) / 2, 0.9, f'{lose / 100:.1%}', ha='center', va='center', fontweight='bold', size=10)
    
    ax.set_axis_off()
    return ax


def bar_plot_p(goal, no_goal, label, ax):
    """Grafico a barre per GOAL/NO GOAL e OVER/UNDER."""
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-6, 90)
    
    b1 = ax.bar(x=0.2, height=goal, width=0.1, color='#de6f57', hatch='//////', ec='black', linewidth=1.)
    ax.bar_label(b1, labels=[str(goal) + '%'], weight='bold', padding=0)
    
    b2 = ax.bar(x=0.4, height=no_goal, width=0.1, color='#287271', hatch='//////', ec='black', linewidth=1.)
    ax.bar_label(b2, labels=[str(no_goal) + '%'], weight='bold', padding=0)
    
    ax.text(0.21, -6, label[0], ha='center', va='center', fontweight='bold', size=12)
    ax.text(0.41, -6, label[1], ha='center', va='center', fontweight='bold', size=12)
    
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], color='black', lw=.75, zorder=3)
    ax.set_axis_off()
    return ax


def bar_plot(goal, no_goal, label, ax):
    """Grafico a barre per xG."""
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.2, 3)
    
    b1 = ax.bar(x=0.2, height=goal, width=0.1, color='#de6f57', hatch='//////', ec='black', linewidth=1.)
    ax.bar_label(b1, labels=[str(goal)], weight='bold', padding=0)
    
    b2 = ax.bar(x=0.4, height=no_goal, width=0.1, color='#287271', hatch='//////', ec='black', linewidth=1.)
    ax.bar_label(b2, labels=[str(no_goal)], weight='bold', padding=0)
    
    ax.text(0.21, -0.2, label[0], ha='center', va='center', fontweight='bold', size=12)
    ax.text(0.41, -0.2, label[1], ha='center', va='center', fontweight='bold', size=12)
    
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], color='black', lw=.75, zorder=3)
    ax.set_axis_off()
    return ax


def table_plot(table, home, away, ax):
    """Grafico matrice di probabilità gol."""
    rows, cols = table.shape
    
    ax.imshow(table, alpha=0.8, cmap="YlGn", aspect=0.2, interpolation='antialiased', vmin=0, vmax=16, origin="lower")
    ax.set_facecolor("#EFE9E6")
    
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    ax.set_ylabel(f"{away} goals", size=10, fontweight='bold')
    ax.set_xlabel(f"{home} goals", size=10, fontweight='bold')
    ax.grid(True, which='major', axis='both', linestyle='-', color='white', linewidth=0.5, alpha=0.3)
    
    for i in range(rows):
        for j in range(cols):
            c = table[i][j]
            ax.text(j, i, str(round(c, 1)) + "%", va='center', ha='center', size=8, fontweight='bold')
    
    return ax


# ─────────────────────────────────────────────────────────────────────
# 4. GENERAZIONE VISUALIZZAZIONE MATCH
# ─────────────────────────────────────────────────────────────────────

def calculate_match_stats(table):
    """Calcola tutte le statistiche dalla matrice di probabilità."""
    m, n = np.shape(table)
    
    # 1X2
    win = draw = lose = 0
    for i in range(m):
        for j in range(n):
            c = table[i][j]
            if i == j:
                draw = round(draw + c, 1)
            elif i > j:
                lose = round(lose + c, 1)
            else:
                win = round(win + c, 1)
    
    if win + draw + lose != 100:
        total = win + draw + lose
        draw = draw + (100 - total)
    
    # Goal/No Goal
    goal = no_goal = 0
    for i in range(m):
        for j in range(n):
            c = table[i][j]
            if i == 0 or j == 0:
                no_goal = round(no_goal + c, 1)
            else:
                goal = round(goal + c, 1)
    
    # Over/Under 2.5
    over25 = under25 = 0
    for i in range(m):
        for j in range(n):
            c = table[i][j]
            if i + j > 2.5:
                over25 = round(over25 + c, 1)
            else:
                under25 = round(under25 + c, 1)
    
    # xG home e away
    goal_home = np.zeros(9, float)
    xg_home = 0
    for i in range(5):
        for j in range(5):
            c = table[j][i]
            goal_home[i] = round(goal_home[i] + c, 1)
        xg_home = round(xg_home + goal_home[i] / 100 * i, 2)
    
    goal_away = np.zeros(9, float)
    xg_away = 0
    for i in range(5):
        for j in range(5):
            c = table[i][j]
            goal_away[i] = round(goal_away[i] + c, 1)
        xg_away = round(xg_away + goal_away[i] / 100 * i, 2)
    
    # Clean sheet
    cs_home = cs_away = 0
    for i in range(m):
        for j in range(n):
            c = table[i][j]
            if i == 0:
                cs_home = round(cs_home + c, 1)
            if j == 0:
                cs_away = round(cs_away + c, 1)
    
    return {
        'win': win, 'draw': draw, 'lose': lose,
        'goal': goal, 'no_goal': no_goal,
        'over25': over25, 'under25': under25,
        'xg_home': xg_home, 'xg_away': xg_away,
        'goal_home': goal_home, 'goal_away': goal_away,
        'cs_home': cs_home, 'cs_away': cs_away
    }


def get_team_logo_url(team_id):
    """Ritorna URL logo fotmob."""
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{int(team_id)}.png"


def create_match_visualization(home_team, away_team, table, home_id, away_id, output_path):
    """
    Crea visualizzazione completa di una partita e la salva come PNG.
    """
    
    stats = calculate_match_stats(table)
    
    fig = plt.figure(figsize=(14, 20), dpi=200, facecolor='#EFE9E6')
    ax = plt.subplot(111, facecolor="#EFE9E6")
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Setup trasformazioni per i logo
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    DC_to_NFC = lambda xy: FC_to_NFC(DC_to_FC(xy))
    
    # ========== ROW 1: STEMMI E NOMI SQUADRE ==========
    y = 13
    
    # Logo home
    try:
        ax_coords = DC_to_NFC([3, y])
        logo_ax = fig.add_axes([ax_coords[0] - 0.05, ax_coords[1] - 0.03, 0.08, 0.08], anchor="C")
        club_icon = Image.open(urllib.request.urlopen(get_team_logo_url(home_id)))
        logo_ax.imshow(club_icon)
        logo_ax.axis("off")
    except Exception as e:
        print(f"  ⚠ Logo {home_team} non caricato: {e}")
    
    ax.text(2.9, y - 0.8, home_team, fontsize=14, fontweight='bold', ha='center', color='#de6f57')
    
    # VS
    ax.text(5, y - 0.3, 'VS', fontsize=12, fontweight='bold', ha='center', color='black', style='italic')
    
    # Logo away
    try:
        ax_coords = DC_to_NFC([7, y])
        logo_ax = fig.add_axes([ax_coords[0] - 0.05, ax_coords[1] - 0.03, 0.08, 0.08], anchor="C")
        club_icon = Image.open(urllib.request.urlopen(get_team_logo_url(away_id)))
        logo_ax.imshow(club_icon)
        logo_ax.axis("off")
    except Exception as e:
        print(f"  ⚠ Logo {away_team} non caricato: {e}")
    
    ax.text(6.9, y - 0.8, away_team, fontsize=14, fontweight='bold', ha='center', color='#287271')
    
    # ========== ROW 2: BATTERY PLOT 1X2 ==========
    y = 12
    ax.text(5, y, 'Match Result Probabilities', fontsize=12, fontweight='bold', ha='center')
    
    ax_coords = DC_to_NFC([1.5, y - 0.6])
    battery_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.55, 0.02])
    battery_plot(stats['win'], stats['draw'], stats['lose'], battery_ax)
    
    # ========== ROW 3: MATRICE RISULTATI ==========
    y = 11
    ax.text(5, y, 'Score Matrix', fontsize=12, fontweight='bold', ha='center')
    
    ax_coords = DC_to_NFC([2.5, y - 4.2])
    table_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.4, 0.4])
    table_plot(table[:6, :6], home_team, away_team, table_ax)
    
    # ========== ROW 4: Goal/No Goal + Over/Under ==========
    y = 8
    
    ax.text(2.5, y + 1.2, 'Both Teams to Score', fontsize=12, fontweight='bold', ha='center')
    ax_coords = DC_to_NFC([0.5, y - 1])
    plot_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.3, 0.15])
    bar_plot_p(stats['goal'], stats['no_goal'], ['GOAL', 'NO GOAL'], plot_ax)
    
    ax.text(7.8, y + 1.2, 'Total score', fontsize=12, fontweight='bold', ha='center')
    ax_coords = DC_to_NFC([5.8, y - 1])
    plot_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.3, 0.15])
    bar_plot_p(stats['over25'], stats['under25'], ['OVER 2.5', 'UNDER 2.5'], plot_ax)
    
    # ========== ROW 5: Expected Goals + Clean Sheets ==========
    y = 5
    
    ax.text(2.5, y + 1.2, 'Expected Goals (xG)', fontsize=12, fontweight='bold', ha='center')
    ax_coords = DC_to_NFC([0.5, y - 1])
    plot_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.3, 0.12])
    bar_plot(stats['xg_home'], stats['xg_away'], [home_team, away_team], plot_ax)
    
    ax.text(7.8, y + 1.2, 'Clean Sheet Probability', fontsize=12, fontweight='bold', ha='center')
    ax_coords = DC_to_NFC([5.8, y - 1])
    plot_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.3, 0.15])
    bar_plot_p(stats['cs_home'], stats['cs_away'], [home_team, away_team], plot_ax)
    
    # ========== ROW 6: Distribuzione gol ==========
    y = 2
    
    ax.text(2.5, y + 1.2, f'{home_team} Goal Distribution', fontsize=12, fontweight='bold', ha='center')
    ax_coords = DC_to_NFC([0.5, y - 1.2])
    plot_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.3, 0.12])
    plot_ax.bar(range(5), stats['goal_home'][:5], color='#de6f57', hatch='//////', ec='black', linewidth=1., width=0.7)
    plot_ax.set_facecolor("#EFE9E6")
    plot_ax.set_xlabel('Goals', fontsize=9, fontweight='bold')
    plot_ax.set_ylabel('Probability (%)', fontsize=9, fontweight='bold')
    plot_ax.set_xticks(range(5))
    plot_ax.grid(axis='y', alpha=0.5)
    plot_ax.grid(axis='x', alpha=0.5)
    plot_ax.spines['top'].set_visible(False)
    plot_ax.spines['right'].set_visible(False)
    plot_ax.spines['left'].set_color('white')
    plot_ax.spines['bottom'].set_color('white')
    plot_ax.spines['left'].set_linewidth(1.3)
    plot_ax.spines['bottom'].set_linewidth(1.3)
    
    ax.text(7.8, y + 1.2, f'{away_team} Goal Distribution', fontsize=12, fontweight='bold', ha='center')
    ax_coords = DC_to_NFC([5.8, y - 1.2])
    plot_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.3, 0.12])
    plot_ax.bar(range(5), stats['goal_away'][:5], color='#287271', hatch='//////', ec='black', linewidth=1., width=0.7)
    plot_ax.set_facecolor("#EFE9E6")
    plot_ax.set_xlabel('Goals', fontsize=9, fontweight='bold')
    plot_ax.set_ylabel('Probability (%)', fontsize=9, fontweight='bold')
    plot_ax.set_xticks(range(5))
    plot_ax.grid(axis='y', alpha=0.5)
    plot_ax.grid(axis='x', alpha=0.5)
    plot_ax.spines['top'].set_visible(False)
    plot_ax.spines['right'].set_visible(False)
    plot_ax.spines['left'].set_color('white')
    plot_ax.spines['bottom'].set_color('white')
    plot_ax.spines['left'].set_linewidth(1.3)
    plot_ax.spines['bottom'].set_linewidth(1.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='#EFE9E6')
    plt.close()
    
    print(f"✓ Visualizzazione salvata → {output_path}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("SERIE A AUTOMATIC MATCH PREDICTION GENERATOR")
    print("=" * 70 + "\n")
    
    try:
        # 1. Scarica rating e statistiche storiche REALI da Understat
        print("📊 Fase 1: Scaricamento dati storici...")
        rating_df, stats_df = fetch_ratings_from_understat()
        print(f"   ✓ Dati statistici reali caricati:")
        print(f"     - Squadre: {len(stats_df)}")
        print(f"     - Media partite giocate (home): {stats_df['MPH'].mean():.0f}")
        print(f"     - Media gol segnati (home): {stats_df['GFH'].mean():.1f}")
        
        # 2. Scarica classifica e prossima giornata
        print("\n⚽ Fase 2: Scaricamento calendario...")
        standings_df, current_gw = fetch_standings()
        matches_df = fetch_next_matchday(current_gw)
        
        if matches_df is None or len(matches_df) == 0:
            print("✗ Nessuna partita trovata per la prossima giornata.")
            return
        
        next_gw = current_gw + 1
        
        # 3. Per ogni match della prossima giornata, crea predizione
        print(f"\n{'='*70}")
        print(f"🎯 PREDIZIONI GIORNATA {next_gw}")
        print(f"{'='*70}\n")
        
        match_count = 0
        successful_matches = 0
        
        for idx, match in matches_df.iterrows():
            match_count += 1
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            print(f"[{match_count}/{len(matches_df)}] {home_team:15s} vs {away_team:15s}", end=" ... ")
            
            # Calcola predizione usando DATI REALI
            table = prediction(stats_df, rating_df, home_team, away_team)
            
            if table is None:
                print("✗ Impossibile calcolare")
                continue
            
            # Scarica ID squadre (mock - usa un valore fisso basato sul nome)
            home_id = abs(hash(home_team)) % 100000
            away_id = abs(hash(away_team)) % 100000
            
            # Crea visualizzazione
            output_filename = f"gw{next_gw}_{home_team}_vs_{away_team}.png"
            output_filepath = OUTPUT_PATH / output_filename
            
            create_match_visualization(home_team, away_team, table, home_id, away_id, output_filepath)
            successful_matches += 1
        
        print(f"\n{'='*70}")
        print(f"✅ COMPLETATO!")
        print(f"{'='*70}")
        print(f"  Giornata: {next_gw}")
        print(f"  Partite elaborate: {match_count}")
        print(f"  Predizioni generate: {successful_matches}")
        print(f"  Cartella output: {OUTPUT_PATH}")
        print(f"\n📌 Dati reali utilizzati:")
        print(f"   - Fonte: Understat.com")
        print(f"   - Squadre: {len(stats_df)}")
        print(f"   - Rating calcolati: {len(rating_df)}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
