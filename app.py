"""
Streamlit-Dashboard: Quantitative Saisonalitätsanalyse
--------------------------------------------------------
Lädt historische Kursdaten via yfinance, berechnet saisonale Renditemuster
(monatlich oder nach Wochentag), führt einen statistischen Signifikanztest
durch und stellt Ergebnisse als Chart + Metrik-Tabelle dar.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime

# ----------------------------------------------------------------------------
# Grundkonfiguration & statische Mappings
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Saisonalitäts-Dashboard", layout="wide")

MONTH_NAMES = {
    1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai", 6: "Juni",
    7: "Juli", 8: "August", 9: "September", 10: "Oktober", 11: "November", 12: "Dezember",
}
WEEKDAY_NAMES = {0: "Montag", 1: "Dienstag", 2: "Mittwoch", 3: "Donnerstag", 4: "Freitag"}

# Muster-Filter: definiert, welche Kalendermonate berücksichtigt werden
# UND liefert gleichzeitig die "natürliche" Reihenfolge für Chart/Tabelle
# (z. B. Nov -> Apr statt numerisch 1 -> 12).
PATTERN_MONTHS = {
    "Ganzes Jahr": list(range(1, 13)),
    "Halloween-Effekt (Nov - Apr)": [11, 12, 1, 2, 3, 4],
    "Sommerloch (Mai - Okt)": [5, 6, 7, 8, 9, 10],
    "Jahresend-Stärke (Okt - Dez)": [10, 11, 12],
}

# ----------------------------------------------------------------------------
# Sidebar: Eingabefelder & Dropdowns
# ----------------------------------------------------------------------------
st.sidebar.header("Einstellungen")

ticker_input = st.sidebar.text_input(
    "Ticker-Symbol",
    value="AAPL",
    help="Aktien z. B. 'AAPL', ETFs z. B. 'SPY', Forex z. B. 'EURUSD=X', Futures z. B. 'GC=F'",
)

years_history = st.sidebar.slider(
    "Historie (Jahre)", min_value=1, max_value=60, value=10
)

analysis_level = st.sidebar.selectbox(
    "Analyse-Ebene",
    ["Monatliche Saisonalität", "Wochentags-Saisonalität"],
)

test_choice = st.sidebar.selectbox(
    "Statistischer Test",
    ["Kruskal-Wallis-Test", "Friedman-Test", "Ljung-Box-Test"],
)

pattern_filter = st.sidebar.selectbox(
    "Muster-Filter",
    list(PATTERN_MONTHS.keys()),
)

st.title("📊 Quantitative Saisonalitätsanalyse")
st.caption(f"{ticker_input.upper()} · {years_history} Jahre Historie · {analysis_level} · Filter: {pattern_filter}")


# ----------------------------------------------------------------------------
# Datenbeschaffung (gecached, damit nicht bei jeder Interaktion neu geladen wird)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Lade Kursdaten von yfinance...")
def load_price_data(ticker: str, years: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - pd.DateOffset(years=years)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return df


# ----------------------------------------------------------------------------
# Renditeberechnung
# ----------------------------------------------------------------------------
def compute_returns(df: pd.DataFrame, level: str) -> pd.Series:
    """
    Berechnet die Renditezeitreihe auf einer durchgehenden (nicht gefilterten)
    Kursreihe, um Verzerrungen durch Lücken an Filter-Grenzen zu vermeiden.
    - Monatlich: Rendite von Monatsende zu Monatsende
    - Wochentag: tägliche Rendite (Close zu Close)
    """
    close = df["Close"].dropna()
    if level == "Monatliche Saisonalität":
        monthly_close = close.resample("ME").last()
        returns = monthly_close.pct_change().dropna()
    else:
        returns = close.pct_change().dropna()
        # Nur Handelstage Mo-Fr behalten (falls vereinzelt Wochenend-Timestamps auftauchen)
        returns = returns[returns.index.weekday < 5]
    return returns


def apply_pattern_filter(returns: pd.Series, pattern: str) -> pd.Series:
    """Filtert die (bereits berechnete) Renditereihe auf die Monate des gewählten Musters."""
    allowed_months = PATTERN_MONTHS[pattern]
    return returns[returns.index.month.isin(allowed_months)]


def get_group_keys(returns: pd.Series, level: str):
    """Liefert (group_keys, order, label_map) passend zur Analyse-Ebene."""
    if level == "Monatliche Saisonalität":
        group_keys = returns.index.month
        order = PATTERN_MONTHS[pattern_filter]  # Reihenfolge entsprechend Muster-Filter
        label_map = MONTH_NAMES
    else:
        group_keys = returns.index.weekday
        order = [0, 1, 2, 3, 4]  # Montag - Freitag, unabhängig vom Muster-Filter
        label_map = WEEKDAY_NAMES
    return np.array(group_keys), order, label_map


# ----------------------------------------------------------------------------
# Statistischer Test
# ----------------------------------------------------------------------------
def run_statistical_test(test_name, returns, group_keys, order, level):
    """Führt den gewählten Test
