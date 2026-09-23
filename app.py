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
        monthly_close = close.resample("M").last()
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
    """Führt den gewählten Test durch und gibt (statistik, p_wert, hinweis) zurück."""
    note = ""
    if test_name == "Kruskal-Wallis-Test":
        # Vergleicht die Verteilungen der Renditen zwischen den Gruppen (Monate/Wochentage)
        groups = [returns[group_keys == k] for k in order if (group_keys == k).any()]
        groups = [g for g in groups if len(g) > 0]
        stat, p = stats.kruskal(*groups)

    elif test_name == "Friedman-Test":
        # Blockdesign: pro Jahr eine durchschnittliche Rendite je Gruppe (verbundene Stichproben)
        df = pd.DataFrame({"return": returns.values, "group": group_keys, "year": returns.index.year})
        pivot = df.pivot_table(index="year", columns="group", values="return", aggfunc="mean")
        pivot = pivot.reindex(columns=[k for k in order if k in pivot.columns]).dropna()
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            raise ValueError(
                "Zu wenig vollständige Jahres-Blöcke für den Friedman-Test "
                "(mindestens 3 Jahre und 3 Gruppen erforderlich)."
            )
        stat, p = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])
        note = f"Basierend auf {pivot.shape[0]} vollständigen Jahres-Blöcken."

    elif test_name == "Ljung-Box-Test":
        # Testet die (gefilterte) Renditereihe auf Autokorrelation / Nicht-Zufälligkeit
        max_lag = 12 if level == "Monatliche Saisonalität" else 10
        lag = max(1, min(max_lag, len(returns) - 1))
        result = acorr_ljungbox(returns, lags=[lag], return_df=True)
        stat = result["lb_stat"].iloc[0]
        p = result["lb_pvalue"].iloc[0]
        note = f"Getestet auf Autokorrelation bis Lag {lag}."

    else:
        raise ValueError("Unbekannter Test")

    return stat, p, note


# ----------------------------------------------------------------------------
# Trading-Metriken-Tabelle
# ----------------------------------------------------------------------------
def calc_metrics_table(returns, group_keys, order, label_map) -> pd.DataFrame:
    records = []
    for key in order:
        subset = returns[group_keys == key]
        if len(subset) == 0:
            continue
        wins = subset[subset > 0]
        losses = subset[subset < 0]
        records.append({
            "Periode": label_map[key],
            "Durchschnittsrendite (%)": subset.mean() * 100,
            "Win Rate (%)": (subset > 0).mean() * 100,
            "Average Win (%)": wins.mean() * 100 if len(wins) > 0 else np.nan,
            "Average Loss (%)": losses.mean() * 100 if len(losses) > 0 else np.nan,
        })
    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# Hauptablauf
# ----------------------------------------------------------------------------
if not ticker_input:
    st.info("Bitte ein Ticker-Symbol eingeben.")
    st.stop()

raw_data = load_price_data(ticker_input.strip().upper(), years_history)

if raw_data.empty or "Close" not in raw_data.columns:
    st.error(f"Keine Daten für Ticker '{ticker_input}' gefunden. Bitte Symbol prüfen.")
    st.stop()

# Renditen berechnen -> Muster-Filter anwenden -> gruppieren
all_returns = compute_returns(raw_data, analysis_level)
filtered_returns = apply_pattern_filter(all_returns, pattern_filter)

if len(filtered_returns) < 5:
    st.error("Zu wenige Datenpunkte nach Filterung für eine aussagekräftige Analyse.")
    st.stop()

group_keys, group_order, label_map = get_group_keys(filtered_returns, analysis_level)

# --- Statistischer Test -------------------------------------------------
st.subheader("Statistischer Signifikanztest")
try:
    stat_value, p_value, test_note = run_statistical_test(
        test_choice, filtered_returns, group_keys, group_order, analysis_level
    )

    col1, col2 = st.columns([2, 1])
    with col2:
        st.metric("Teststatistik", f"{stat_value:.4f}")
        st.metric("p-Wert", f"{p_value:.4f}")

    with col1:
        # Ampelsystem: p < 0.05 -> signifikant (grün), sonst nicht signifikant (rot)
        if p_value < 0.05:
            st.success(
                f"✅ {test_choice}: p-Wert = {p_value:.4f} (< 0.05) — "
                f"statistisch signifikante Saisonalität erkennbar."
            )
        else:
            st.error(
                f"❌ {test_choice}: p-Wert = {p_value:.4f} (≥ 0.05) — "
                f"keine statistisch signifikante Saisonalität erkennbar."
            )
        if test_note:
            st.caption(test_note)

except ValueError as e:
    st.warning(f"Test konnte nicht durchgeführt werden: {e}")

# --- Balkendiagramm -------------------------------------------------------
st.subheader("Durchschnittliche Rendite pro Periode")

metrics_df = calc_metrics_table(filtered_returns, group_keys, group_order, label_map)

bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in metrics_df["Durchschnittsrendite (%)"]]

fig = go.Figure(
    data=[
        go.Bar(
            x=metrics_df["Periode"],
            y=metrics_df["Durchschnittsrendite (%)"],
            marker_color=bar_colors,
            text=metrics_df["Durchschnittsrendite (%)"].round(2),
            texttemplate="%{text}%",
            textposition="outside",
        )
    ]
)
fig.update_layout(
    xaxis_title="Periode",
    yaxis_title="Durchschnittliche Rendite (%)",
    template="plotly_white",
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

# --- Trading-Metriken-Tabelle ---------------------------------------------
st.subheader("Trading-Metriken je Periode")
display_df = metrics_df.copy()
for col in ["Durchschnittsrendite (%)", "Win Rate (%)", "Average Win (%)", "Average Loss (%)"]:
    display_df[col] = display_df[col].round(2)

st.dataframe(display_df.set_index("Periode"), use_container_width=True)

st.caption(
    "Hinweis: Alle Werte basieren auf historischen Daten und stellen keine Anlageberatung dar."
)
