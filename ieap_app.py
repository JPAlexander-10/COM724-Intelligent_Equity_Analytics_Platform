import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IEAP — Intelligent Equity Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0A0A0A;
        color: #F0F0F0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222222;
    }

    /* Hero banner */
    .ieap-hero {
        background: #F8C300;
        padding: 28px 36px;
        border-radius: 4px;
        margin-bottom: 28px;
    }
    .ieap-hero h1 {
        color: #000000;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .ieap-hero p {
        color: #1a1a1a;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: #141414;
        border: 1px solid #222222;
        border-radius: 4px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .metric-label {
        color: #888888;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F8C300;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'DM Mono', monospace;
        line-height: 1;
    }
    .metric-sub {
        color: #555555;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* Signal badges */
    .signal-buy {
        background: #00C853;
        color: #000;
        padding: 3px 10px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 0.8rem;
        font-family: 'DM Mono', monospace;
    }
    .signal-sell {
        background: #FF1744;
        color: #fff;
        padding: 3px 10px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 0.8rem;
        font-family: 'DM Mono', monospace;
    }
    .signal-hold {
        background: #F8C300;
        color: #000;
        padding: 3px 10px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 0.8rem;
        font-family: 'DM Mono', monospace;
    }

    /* Section headers */
    .section-header {
        color: #F8C300;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #222;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111111;
        border-bottom: 1px solid #222222;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #666666;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 10px 20px;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #F8C300 !important;
        border-bottom: 2px solid #F8C300 !important;
        background-color: transparent !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] {
        padding-top: 8px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Table styling */
    .stDataFrame {
        background: #141414;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #141414;
        border: 1px solid #333;
        color: #F0F0F0;
    }

    /* Selectbox dropdown options — taxi yellow */
    [data-baseweb="option"] {
        color: #F8C300 !important;
        background-color: #141414 !important;
    }
    [data-baseweb="option"]:hover {
        background-color: #1E1E1E !important;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
STOCKS = ['ISRG', 'AMAT', 'TMUS', 'TSLA']
SECTORS = {
    'ISRG': 'Medical Robotics',
    'AMAT': 'Semiconductor Equipment',
    'TMUS': 'Telecommunications',
    'TSLA': 'Electric Vehicles / Tech'
}
COLOURS = {
    'ISRG': '#4FC3F7',
    'AMAT': '#FF8A65',
    'TMUS': '#81C784',
    'TSLA': '#EF5350'
}
CLUSTERS = {
    'ISRG': 'Cluster 0 — Large Cap Tech & Growth',
    'AMAT': 'Cluster 1 — Semiconductor & AI Hardware',
    'TMUS': 'Cluster 2 — Defensive & Steady Growth',
    'TSLA': 'Cluster 3 — High Volatility / Speculative'
}

DATA_DIR = Path("ieap_data")

FEATURE_LABELS = {
    'returns': 'Daily Returns',
    'ma20': '20-Day MA',
    'ma50': '50-Day MA',
    'price_ma20_ratio': 'Price / MA20',
    'price_ma50_ratio': 'Price / MA50',
    'ma20_ma50_ratio': 'MA20 / MA50',
    'volatility_30': '30-Day Volatility',
    'rsi': 'RSI (14)',
    'macd': 'MACD',
    'macd_signal': 'MACD Signal',
    'macd_hist': 'MACD Histogram'
}

PLOTLY_THEME = dict(
    paper_bgcolor='#0A0A0A',
    plot_bgcolor='#0F0F0F',
    font=dict(family='DM Sans', color='#AAAAAA', size=11),
    xaxis=dict(gridcolor='#1A1A1A', linecolor='#222222', tickcolor='#333333'),
    yaxis=dict(gridcolor='#1A1A1A', linecolor='#222222', tickcolor='#333333'),
    margin=dict(l=40, r=20, t=40, b=40)
)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_prices():
    df = pd.read_csv(DATA_DIR / 'ieap_prices.csv', index_col=0, parse_dates=True)
    return df

@st.cache_data
def load_returns():
    df = pd.read_csv(DATA_DIR / 'ieap_returns.csv', index_col=0, parse_dates=True)
    return df

@st.cache_data
def load_pearson():
    return pd.read_csv(DATA_DIR / 'ieap_pearson_corr.csv', index_col=0)

@st.cache_data
def load_spearman():
    return pd.read_csv(DATA_DIR / 'ieap_spearman_corr.csv', index_col=0)

@st.cache_data
def load_arima_results():
    return pd.read_csv(DATA_DIR / 'ieap_arima_results.csv')

@st.cache_data
def load_lstm_results():
    return pd.read_csv(DATA_DIR / 'ieap_lstm_results.csv')

@st.cache_data
def load_lstm_forecast(stock):
    df = pd.read_csv(DATA_DIR / f'ieap_lstm_forecast_{stock}.csv', parse_dates=['Date'])
    return df.set_index('Date')

@st.cache_data
def load_signals(stock):
    df = pd.read_csv(DATA_DIR / f'ieap_signals_{stock}.csv', parse_dates=['Date'])
    return df.set_index('Date')

@st.cache_data
def load_signal_summary():
    return pd.read_csv(DATA_DIR / 'ieap_signal_summary.csv')

@st.cache_data
def load_scenario():
    return pd.read_csv(DATA_DIR / 'ieap_scenario_results.csv')

@st.cache_data
def load_importance(stock):
    return pd.read_csv(DATA_DIR / f'ieap_importance_{stock}.csv', index_col=0)

@st.cache_resource
def load_rf_model(stock):
    path = DATA_DIR / 'ieap_models' / f'rf_{stock}.pkl'
    if path.exists():
        return joblib.load(path)
    return None

def check_data():
    required = ['ieap_prices.csv', 'ieap_returns.csv', 'ieap_pearson_corr.csv',
                'ieap_spearman_corr.csv', 'ieap_lstm_results.csv', 'ieap_arima_results.csv',
                'ieap_signal_summary.csv', 'ieap_scenario_results.csv']
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    return missing

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def build_features(prices, returns):
    df = pd.DataFrame(index=prices.index)
    df['returns'] = returns
    df['ma20'] = prices.rolling(20).mean()
    df['ma50'] = prices.rolling(50).mean()
    df['price_ma20_ratio'] = prices / df['ma20']
    df['price_ma50_ratio'] = prices / df['ma50']
    df['ma20_ma50_ratio'] = df['ma20'] / df['ma50']
    df['volatility_30'] = returns.rolling(30).std() * np.sqrt(252)
    df['rsi'] = compute_rsi(prices)
    macd, sig, hist = compute_macd(prices)
    df['macd'] = macd
    df['macd_signal'] = sig
    df['macd_hist'] = hist
    return df.dropna()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:#F8C300;padding:16px 18px;border-radius:4px;margin-bottom:20px;">
        <div style="font-size:1.1rem;font-weight:700;color:#000;letter-spacing:-0.3px;">IEAP</div>
        <div style="font-size:0.7rem;color:#333;font-weight:500;letter-spacing:0.5px;">INTELLIGENT EQUITY ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">PORTFOLIO</div>', unsafe_allow_html=True)
    selected_stock = st.selectbox(
        "Select Stock",
        STOCKS,
        format_func=lambda x: f"{x} — {SECTORS[x]}"
    )

    st.markdown('<div class="section-header">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio(
        "View",
        ["Stock Analysis", "Portfolio Overview"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="color:#FFFFFF;font-size:0.7rem;line-height:1.6;">
        <div style="color:#FFFFFF;font-weight:600;margin-bottom:6px;">DATA PERIOD</div>
        Jan 2016 — Jan 2026<br>
        <div style="color:#FFFFFF;font-weight:600;margin-top:10px;margin-bottom:6px;">UNIVERSE</div>
        Top 30 NASDAQ-100<br>
        <div style="color:#FFFFFF;font-weight:600;margin-top:10px;margin-bottom:6px;">PORTFOLIO</div>
        ISRG · AMAT · TMUS · TSLA
    </div>
    """, unsafe_allow_html=True)

# ── HERO BANNER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="ieap-hero">
    <h1>Intelligent Equity Analytics Platform</h1>
    <p>SOLiGence Decision Support System &nbsp;·&nbsp; COM724 Applied AI in Business &nbsp;·&nbsp; Jean-Pierre Alexander</p>
</div>
""", unsafe_allow_html=True)

# ── CHECK DATA ────────────────────────────────────────────────────────────────
missing = check_data()
if missing:
    st.error(f"⚠️ Missing data files in `ieap_data/` folder: {', '.join(missing)}")
    st.info("Please run the export cells in your Colab notebook and place all files in the `ieap_data` folder.")
    st.stop()

# Load all data
prices = load_prices()
returns = load_returns()
pearson = load_pearson()
spearman = load_spearman()
arima_res = load_arima_results()
lstm_res = load_lstm_results()
signal_summary = load_signal_summary()
scenario = load_scenario()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "Stock Analysis":

    stock = selected_stock
    colour = COLOURS[stock]
    stock_prices = prices[stock]
    stock_returns = returns[stock]

    # ── STOCK HEADER ─────────────────────────────────────────────────────────
    total_ret = ((stock_prices.iloc[-1] / stock_prices.iloc[0]) - 1) * 100
    ann_vol = stock_returns.std() * np.sqrt(252) * 100
    rolling_peak = stock_prices.cummax()
    max_dd = ((stock_prices - rolling_peak) / rolling_peak * 100).min()
    current_price = stock_prices.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-sub">{stock} · {SECTORS[stock]}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        sign = "+" if total_ret > 0 else ""
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">10-Year Return</div>
            <div class="metric-value">{sign}{total_ret:.1f}%</div>
            <div class="metric-sub">Jan 2016 → Jan 2026</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Ann. Volatility</div>
            <div class="metric-value">{ann_vol:.1f}%</div>
            <div class="metric-sub">Annualised std deviation</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value">{max_dd:.1f}%</div>
            <div class="metric-sub">Peak to trough</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Forecasting",
        "Trading Signals",
        "Scenario Analysis"
    ])

    # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([3, 1])

        with col_a:
            # Price history chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_prices.index, y=stock_prices.values,
                name=stock, line=dict(color=colour, width=1.8),
                fill='tozeroy', fillcolor=f'rgba({int(colour[1:3],16)},{int(colour[3:5],16)},{int(colour[5:7],16)},0.06)'
            ))

            events = {
                '2018-12-24': '2018 Correction',
                '2020-03-23': 'COVID Crash',
                '2022-01-03': '2022 Bear Market',
                '2023-01-01': 'AI Bull Run'
            }
            for date, label in events.items():
                fig.add_vline(x=date, line_dash="dot", line_color="#333333", line_width=1)
                fig.add_annotation(x=date, y=0, text=label, textangle=-90,
                                   font=dict(size=8, color="#444444"), showarrow=False,
                                   yref="paper", yanchor="bottom")

            fig.update_layout(**PLOTLY_THEME, title=f"{stock} Price History (2016–2026)",
                              title_font=dict(color='#F0F0F0', size=13),
                              height=350, showlegend=False,
                              yaxis_tickprefix="$", yaxis_tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown chart
            drawdown = ((stock_prices - rolling_peak) / rolling_peak * 100)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=drawdown.index, y=drawdown.values,
                fill='tozeroy', fillcolor='rgba(239,83,80,0.15)',
                line=dict(color='#EF5350', width=1.2), name='Drawdown'
            ))
            fig2.update_layout(**PLOTLY_THEME, title="Drawdown from Peak (%)",
                               title_font=dict(color='#F0F0F0', size=13),
                               height=220, showlegend=False,
                               yaxis_ticksuffix="%")
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            # Stats table
            from scipy import stats as scipy_stats
            skew = scipy_stats.skew(stock_returns.dropna())
            kurt = scipy_stats.kurtosis(stock_returns.dropna())
            best_day = stock_returns.max() * 100
            worst_day = stock_returns.min() * 100

            st.markdown('<div class="section-header">STATISTICS</div>', unsafe_allow_html=True)
            stats_data = {
                'Metric': ['Start Price', 'End Price', 'Best Day', 'Worst Day',
                           'Skewness', 'Kurtosis', 'Cluster', 'Sector'],
                'Value': [
                    f"${stock_prices.iloc[0]:.2f}",
                    f"${stock_prices.iloc[-1]:.2f}",
                    f"+{best_day:.2f}%",
                    f"{worst_day:.2f}%",
                    f"{skew:.3f}",
                    f"{kurt:.3f}",
                    CLUSTERS[stock].split('—')[0].strip(),
                    SECTORS[stock]
                ]
            }
            st.dataframe(
                pd.DataFrame(stats_data).set_index('Metric'),
                use_container_width=True, height=320
            )

    # ── TAB 2: FORECASTING ────────────────────────────────────────────────────
    with tab2:
        arima_match = arima_res[arima_res['Stock'] == stock]
        lstm_match = lstm_res[lstm_res['Stock'] == stock]

        if arima_match.empty or lstm_match.empty:
            st.warning(f"Forecasting results not found for {stock}. Ensure ARIMA and LSTM result files are in ieap_data/")
        else:
            arima_row = arima_match.iloc[0]
            lstm_row = lstm_match.iloc[0]
            arima_mape = arima_row['MAPE (%)']
            improvement = ((arima_mape - lstm_row['MAPE (%)']) / arima_mape * 100) if arima_mape != 0 else 0.0

            # Metric row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">ARIMA MAPE</div>
                    <div class="metric-value" style="color:#EF5350">{arima_mape:.2f}%</div>
                    <div class="metric-sub">Baseline model</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">LSTM MAPE</div>
                    <div class="metric-value" style="color:#81C784">{lstm_row['MAPE (%)']:.2f}%</div>
                    <div class="metric-sub">Advanced model</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Improvement</div>
                    <div class="metric-value">{improvement:.0f}%</div>
                    <div class="metric-sub">LSTM over ARIMA</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Winner</div>
                    <div class="metric-value" style="color:#81C784;font-size:1.4rem">LSTM</div>
                    <div class="metric-sub">All 3 metrics</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # LSTM forecast chart
            try:
                lstm_forecast = load_lstm_forecast(stock)
                test_prices = stock_prices[stock_prices.index >= lstm_forecast.index[0]]

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=test_prices.index, y=test_prices.values,
                    name='Actual Price', line=dict(color=colour, width=2)
                ))
                fig3.add_trace(go.Scatter(
                    x=lstm_forecast.index, y=lstm_forecast['Forecast'].values,
                    name='LSTM Forecast', line=dict(color='#F8C300', width=2, dash='dash')
                ))
                fig3.update_layout(**PLOTLY_THEME,
                                   title=f"{stock} — LSTM Forecast vs Actual (2024–2025)",
                                   title_font=dict(color='#F0F0F0', size=13),
                                   height=380, yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                                   legend=dict(bgcolor='#141414', bordercolor='#333'))
                st.plotly_chart(fig3, use_container_width=True)
            except Exception:
                st.info("LSTM forecast file not found. Ensure ieap_lstm_forecast files are in ieap_data/")

            # Model comparison bar chart
            metrics = ['RMSE', 'MAE', 'MAPE (%)']
            arima_vals = [arima_row[m] for m in metrics]
            lstm_vals = [lstm_row[m] for m in metrics]

            fig4 = go.Figure()
            fig4.add_trace(go.Bar(name='ARIMA', x=metrics, y=arima_vals,
                                  marker_color='#EF5350', opacity=0.85))
            fig4.add_trace(go.Bar(name='LSTM', x=metrics, y=lstm_vals,
                                  marker_color='#81C784', opacity=0.85))
            fig4.update_layout(**PLOTLY_THEME,
                               title="ARIMA vs LSTM Error Metrics (Lower = Better)",
                               title_font=dict(color='#F0F0F0', size=13),
                               barmode='group', height=300,
                               legend=dict(bgcolor='#141414', bordercolor='#333'))
            st.plotly_chart(fig4, use_container_width=True)

    # ── TAB 3: TRADING SIGNALS ────────────────────────────────────────────────
    with tab3:
        try:
            signals_df = load_signals(stock)
            sig_summary = signal_summary[signal_summary['Stock'] == stock].iloc[0]

            # Signal counts
            buy_pct = sig_summary['BUY %']
            hold_pct = sig_summary['HOLD %']
            sell_pct = sig_summary['SELL %']
            total_signals = sig_summary['BUY signals'] + sig_summary['HOLD signals'] + sig_summary['SELL signals']

            # Latest signal
            latest_signal_code = int(signals_df['Signal'].iloc[-1])
            latest_signal = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}.get(latest_signal_code, 'HOLD')
            signal_badge = {
                'BUY': '<span class="signal-buy">BUY</span>',
                'HOLD': '<span class="signal-hold">HOLD</span>',
                'SELL': '<span class="signal-sell">SELL</span>'
            }[latest_signal]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Latest Signal</div>
                    <div style="margin-top:8px">{signal_badge}</div>
                    <div class="metric-sub">{signals_df.index[-1].strftime('%d %b %Y')}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">BUY Days</div>
                    <div class="metric-value" style="color:#00C853">{buy_pct:.1f}%</div>
                    <div class="metric-sub">{sig_summary['BUY signals']} of {total_signals} days</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">HOLD Days</div>
                    <div class="metric-value" style="color:#F8C300">{hold_pct:.1f}%</div>
                    <div class="metric-sub">{sig_summary['HOLD signals']} of {total_signals} days</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">SELL Days</div>
                    <div class="metric-value" style="color:#FF1744">{sell_pct:.1f}%</div>
                    <div class="metric-sub">{sig_summary['SELL signals']} of {total_signals} days</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Signal chart
            buy_mask = signals_df['Signal'] == 2
            sell_mask = signals_df['Signal'] == 0

            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=signals_df.index, y=signals_df['Price'],
                name='Price', line=dict(color=colour, width=1.5)
            ))
            if buy_mask.any():
                fig5.add_trace(go.Scatter(
                    x=signals_df.index[buy_mask],
                    y=signals_df['Price'][buy_mask],
                    mode='markers', name=f'BUY ({buy_mask.sum()})',
                    marker=dict(symbol='triangle-up', size=8, color='#00C853',
                                line=dict(color='#ffffff', width=0.5))
                ))
            if sell_mask.any():
                fig5.add_trace(go.Scatter(
                    x=signals_df.index[sell_mask],
                    y=signals_df['Price'][sell_mask],
                    mode='markers', name=f'SELL ({sell_mask.sum()})',
                    marker=dict(symbol='triangle-down', size=8, color='#FF1744',
                                line=dict(color='#ffffff', width=0.5))
                ))

            fig5.update_layout(**PLOTLY_THEME,
                               title=f"{stock} — Trading Signals (2024–2025)",
                               title_font=dict(color='#F0F0F0', size=13),
                               height=420, yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                               legend=dict(bgcolor='#141414', bordercolor='#333'))
            st.plotly_chart(fig5, use_container_width=True)

            # Feature importance
            try:
                importance = load_importance(stock)
                importance.index = [FEATURE_LABELS.get(i, i) for i in importance.index]
                importance = importance.sort_values('Importance')

                fig6 = go.Figure(go.Bar(
                    x=importance['Importance'], y=importance.index,
                    orientation='h', marker_color=colour, opacity=0.85
                ))
                fig6.update_layout(**PLOTLY_THEME,
                                   title="Feature Importance — What Drives the Signal",
                                   title_font=dict(color='#F0F0F0', size=13),
                                   height=350)
                st.plotly_chart(fig6, use_container_width=True)
            except Exception:
                pass

        except Exception as e:
            st.info(f"Signal data not available. Ensure ieap_signals_{stock}.csv is in ieap_data/")

    # ── TAB 4: SCENARIO ANALYSIS ──────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">WHAT-IF SCENARIO SIMULATOR</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#666;font-size:0.82rem;margin-bottom:20px;">
        Adjust the market conditions below to see just how the Random Forest trading signals respond in real time!
        \nThe model will then apply your modifications to the test period feature set and recalculates Buy/Hold/Sell distributions.
        </div>
        """, unsafe_allow_html=True)

        col_sliders, col_results = st.columns([1, 2])

        with col_sliders:
            st.markdown('<div class="section-header">MARKET CONDITIONS</div>', unsafe_allow_html=True)

            vol_multiplier = st.slider(
                "Volatility multiplier",
                min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                help="1.0 = baseline. 2.0 = double the volatility"
            )
            price_drop = st.slider(
                "Price level adjustment (%)",
                min_value=-30, max_value=30, value=0, step=1,
                format="%d%%",
                help="Simulate a price drop or rise relative to moving averages"
            )
            rsi_override = st.slider(
                "RSI override",
                min_value=10, max_value=90, value=50, step=1,
                help="RSI below 30 = oversold (bullish), above 70 = overbought (bearish)"
            )
            macd_override = st.slider(
                "MACD histogram",
                min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
                help="Positive = bullish momentum, negative = bearish"
            )

            run_scenario = st.button("Run Scenario", use_container_width=True)

        with col_results:
            # Always show pre-saved baseline scenarios
            stock_scenarios = scenario[scenario['Stock'] == stock].copy()

            if not stock_scenarios.empty:
                fig7 = go.Figure()
                scenarios_list = stock_scenarios['Scenario'].tolist()
                buy_vals = stock_scenarios['BUY %'].tolist()
                hold_vals = stock_scenarios['HOLD %'].tolist()
                sell_vals = stock_scenarios['SELL %'].tolist()

                fig7.add_trace(go.Bar(name='BUY', x=scenarios_list, y=buy_vals,
                                      marker_color='#00C853', opacity=0.85))
                fig7.add_trace(go.Bar(name='HOLD', x=scenarios_list, y=hold_vals,
                                      marker_color='#F8C300', opacity=0.85))
                fig7.add_trace(go.Bar(name='SELL', x=scenarios_list, y=sell_vals,
                                      marker_color='#EF5350', opacity=0.85))

                fig7.update_layout(**PLOTLY_THEME,
                                   title="Pre-Defined Scenario Results",
                                   title_font=dict(color='#F0F0F0', size=13),
                                   barmode='group', height=320,
                                   legend=dict(bgcolor='#141414', bordercolor='#333'),
                                   xaxis_tickangle=-20)
                st.plotly_chart(fig7, use_container_width=True)

            # Live scenario runner
            if run_scenario:
                rf_model = load_rf_model(stock)
                if rf_model is not None:
                    try:
                        stock_prices_col = prices[stock]
                        stock_returns_col = returns[stock]
                        features = build_features(stock_prices_col, stock_returns_col)

                        split_date = pd.Timestamp('2023-12-29')
                        test_features = features[features.index >= split_date].copy()

                        train_cols = ['returns', 'ma20', 'ma50', 'price_ma20_ratio',
                                      'price_ma50_ratio', 'ma20_ma50_ratio',
                                      'volatility_30', 'rsi', 'macd', 'macd_signal', 'macd_hist']

                        available_cols = [c for c in train_cols if c in test_features.columns]
                        X_scenario = test_features[available_cols].copy()

                        # Apply modifications
                        if 'volatility_30' in X_scenario.columns:
                            X_scenario['volatility_30'] *= vol_multiplier
                        if price_drop != 0:
                            for col in ['price_ma20_ratio', 'price_ma50_ratio']:
                                if col in X_scenario.columns:
                                    X_scenario[col] += (price_drop / 100)
                        if 'rsi' in X_scenario.columns:
                            X_scenario['rsi'] = rsi_override
                        if 'macd_hist' in X_scenario.columns:
                            X_scenario['macd_hist'] = macd_override

                        preds = rf_model.predict(X_scenario)
                        total = len(preds)
                        buy_pct_live = (preds == 2).sum() / total * 100
                        hold_pct_live = (preds == 1).sum() / total * 100
                        sell_pct_live = (preds == 0).sum() / total * 100

                        st.markdown('<div class="section-header">LIVE SCENARIO RESULT</div>', unsafe_allow_html=True)
                        r1, r2, r3 = st.columns(3)
                        with r1:
                            st.markdown(f"""<div class="metric-card" style="border-color:#00C853">
                                <div class="metric-label">BUY</div>
                                <div class="metric-value" style="color:#00C853">{buy_pct_live:.1f}%</div>
                            </div>""", unsafe_allow_html=True)
                        with r2:
                            st.markdown(f"""<div class="metric-card" style="border-color:#F8C300">
                                <div class="metric-label">HOLD</div>
                                <div class="metric-value" style="color:#F8C300">{hold_pct_live:.1f}%</div>
                            </div>""", unsafe_allow_html=True)
                        with r3:
                            st.markdown(f"""<div class="metric-card" style="border-color:#EF5350">
                                <div class="metric-label">SELL</div>
                                <div class="metric-value" style="color:#EF5350">{sell_pct_live:.1f}%</div>
                            </div>""", unsafe_allow_html=True)

                        # Baseline comparison
                        baseline = stock_scenarios[stock_scenarios['Scenario'] == 'Baseline']
                        if not baseline.empty:
                            base_buy = baseline['BUY %'].values[0]
                            diff = buy_pct_live - base_buy
                            direction = "(+)" if diff > 0 else "(-)"
                            st.markdown(f"""
                            <div style="color:#666;font-size:0.8rem;margin-top:8px;padding:10px 14px;background:#141414;border-radius:4px;border:1px solid #222;">
                                vs Baseline: BUY signals {direction} {abs(diff):.1f} percentage points
                            </div>""", unsafe_allow_html=True)

                    except Exception as ex:
                        st.error(f"Could not run live scenario: {ex}")
                else:
                    st.warning("Trained model files not found in ieap_data/ieap_models/. Live scenario requires the .pkl model files.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.markdown('<div class="section-header">PORTFOLIO OVERVIEW — ISRG · AMAT · TMUS · TSLA</div>', unsafe_allow_html=True)

    # ── PORTFOLIO METRICS ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    for i, (stock, col) in enumerate(zip(STOCKS, [col1, col2, col3, col4])):
        sp = prices[stock]
        ret = ((sp.iloc[-1] / sp.iloc[0]) - 1) * 100
        with col:
            st.markdown(f"""<div class="metric-card" style="border-left:3px solid {COLOURS[stock]}">
                <div class="metric-label">{stock}</div>
                <div class="metric-value" style="color:{COLOURS[stock]};font-size:1.5rem">{ret:.0f}%</div>
                <div class="metric-sub">{SECTORS[stock]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Normalised performance
        normalised = (prices / prices.iloc[0]) * 100
        fig8 = go.Figure()
        for stock in STOCKS:
            total_ret = ((prices[stock].iloc[-1] / prices[stock].iloc[0]) - 1) * 100
            fig8.add_trace(go.Scatter(
                x=normalised.index, y=normalised[stock],
                name=f"{stock} ({total_ret:.0f}%)",
                line=dict(color=COLOURS[stock], width=2)
            ))
        fig8.add_hline(y=100, line_dash="dot", line_color="#333333", line_width=1)
        fig8.update_layout(**PLOTLY_THEME,
                           title="Normalised Price Performance (Base = 100)",
                           title_font=dict(color='#F0F0F0', size=13),
                           height=380, yaxis_tickformat=",.0f",
                           legend=dict(bgcolor='#141414', bordercolor='#333'))
        st.plotly_chart(fig8, use_container_width=True)

        # Correlation heatmap
        fig9 = go.Figure(data=go.Heatmap(
            z=pearson.values,
            x=pearson.columns,
            y=pearson.index,
            colorscale='RdYlGn',
            zmin=-1, zmax=1,
            text=np.round(pearson.values, 3),
            texttemplate="%{text}",
            textfont=dict(size=13, color='black', family='DM Mono'),
            showscale=True
        ))
        fig9.update_layout(**PLOTLY_THEME,
                           title="Pearson Correlation Matrix",
                           title_font=dict(color='#F0F0F0', size=13),
                           height=320)
        st.plotly_chart(fig9, use_container_width=True)

    with col_right:
        # Rolling volatility
        fig10 = go.Figure()
        for stock in STOCKS:
            rolling_vol = returns[stock].rolling(30).std() * np.sqrt(252) * 100
            fig10.add_trace(go.Scatter(
                x=rolling_vol.index, y=rolling_vol,
                name=stock, line=dict(color=COLOURS[stock], width=1.5)
            ))
        fig10.update_layout(**PLOTLY_THEME,
                            title="Rolling 30-Day Annualised Volatility (%)",
                            title_font=dict(color='#F0F0F0', size=13),
                            height=380, yaxis_ticksuffix="%",
                            legend=dict(bgcolor='#141414', bordercolor='#333'))
        st.plotly_chart(fig10, use_container_width=True)

        # Model comparison across all stocks
        fig11 = go.Figure()
        fig11.add_trace(go.Bar(
            name='ARIMA MAPE', x=arima_res['Stock'], y=arima_res['MAPE (%)'],
            marker_color='#EF5350', opacity=0.85
        ))
        fig11.add_trace(go.Bar(
            name='LSTM MAPE', x=lstm_res['Stock'], y=lstm_res['MAPE (%)'],
            marker_color='#81C784', opacity=0.85
        ))
        fig11.update_layout(**PLOTLY_THEME,
                            title="ARIMA vs LSTM Forecast Error — All Stocks",
                            title_font=dict(color='#F0F0F0', size=13),
                            barmode='group', height=320,
                            legend=dict(bgcolor='#141414', bordercolor='#333'))
        st.plotly_chart(fig11, use_container_width=True)

    # ── PORTFOLIO SUMMARY TABLE ───────────────────────────────────────────────
    st.markdown('<div class="section-header">PORTFOLIO SUMMARY TABLE</div>', unsafe_allow_html=True)

    summary_rows = []
    for stock in STOCKS:
        sp = prices[stock]
        sr = returns[stock]
        total_ret = ((sp.iloc[-1] / sp.iloc[0]) - 1) * 100
        ann_vol = sr.std() * np.sqrt(252) * 100
        rolling_peak = sp.cummax()
        max_dd = ((sp - rolling_peak) / rolling_peak * 100).min()
        ann_ret = sr.mean() * 252 * 100
        sharpe = ann_ret / ann_vol

        arima_row = arima_res[arima_res['Stock'] == stock]
        lstm_row_data = lstm_res[lstm_res['Stock'] == stock]
        arima_mape = arima_row['MAPE (%)'].values[0] if not arima_row.empty else 'N/A'
        lstm_mape = lstm_row_data['MAPE (%)'].values[0] if not lstm_row_data.empty else 'N/A'

        sig_row = signal_summary[signal_summary['Stock'] == stock]
        dominant = 'N/A'
        if not sig_row.empty:
            r = sig_row.iloc[0]
            dominant = max([('BUY', r['BUY %']), ('HOLD', r['HOLD %']), ('SELL', r['SELL %'])], key=lambda x: x[1])[0]

        summary_rows.append({
            'Stock': stock,
            'Sector': SECTORS[stock],
            '10yr Return': f"{total_ret:.1f}%",
            'Ann. Volatility': f"{ann_vol:.1f}%",
            'Max Drawdown': f"{max_dd:.1f}%",
            'Sharpe (approx)': f"{sharpe:.3f}",
            'ARIMA MAPE': f"{arima_mape:.2f}%" if isinstance(arima_mape, float) else arima_mape,
            'LSTM MAPE': f"{lstm_mape:.2f}%" if isinstance(lstm_mape, float) else lstm_mape,
            'Dominant Signal': dominant
        })

    summary_df = pd.DataFrame(summary_rows)
    styled_summary = summary_df.style.applymap(
        lambda v: 'color: #F8C300; font-weight: 600' if v in STOCKS else '',
        subset=['Stock']
    )
    st.dataframe(styled_summary, use_container_width=True, height=220, hide_index=True)

    # ── PAIRWISE CORRELATION SUMMARY ──────────────────────────────────────────
    st.markdown('<div class="section-header">PAIRWISE CORRELATION SUMMARY</div>', unsafe_allow_html=True)

    corr_rows = []
    for i in range(len(STOCKS)):
        for j in range(i+1, len(STOCKS)):
            s1, s2 = STOCKS[i], STOCKS[j]
            p_val = pearson.loc[s1, s2]
            sp_val = spearman.loc[s1, s2]
            if p_val >= 0.4:
                implication = "Moderate — some diversification benefit"
            else:
                implication = "Low — strong diversification benefit"
            corr_rows.append({'Pair': f"{s1} vs {s2}", 'Pearson': f"{p_val:.4f}",
                               'Spearman': f"{sp_val:.4f}", 'Implication': implication})

    corr_df = pd.DataFrame(corr_rows)
    all_pairs = [f"{STOCKS[i]} vs {STOCKS[j]}" for i in range(len(STOCKS)) for j in range(i + 1, len(STOCKS))]
    styled_corr = corr_df.style.applymap(
        lambda v: 'color: #F8C300; font-weight: 600' if v in all_pairs else '',
        subset=['Pair']
    )
    st.dataframe(styled_corr, use_container_width=True, height=250, hide_index=True)
