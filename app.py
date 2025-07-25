import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

st.set_page_config(page_title="AI Stock Screener (NSE)", layout="wide")
st.title("📈 AI Stock Screener – Fundamentals + Technicals + Score (Buy / Hold / Sell)")

@st.cache_data(ttl=900)
def get_history(symbol, period="2y", interval="1d"):
    return yf.download(symbol, period=period, interval=interval, auto_adjust=True)

@st.cache_data(ttl=900)
def get_info(symbol):
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# --- Fixed RSI ---
def rsi(series, period=14):
    series = series.astype(float)
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def pct(a, low, high, reverse=False):
    if a is None or np.isnan(a):
        return 0
    a = max(min(a, high), low)
    score = (a - low) / (high - low) * 100
    return 100 - score if reverse else score

def classify(score):
    if score >= 70:
        return "✅ Buy"
    elif score >= 40:
        return "🤝 Hold"
    else:
        return "❌ Sell"

def analyze_symbol(symbol):
    info = get_info(symbol)
    hist = get_history(symbol)

    if hist.empty:
        return None

    close = hist["Close"]
    last_price = close.iloc[-1]

    # --- Technical analysis (safe checks) ---
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    rsi14 = rsi(close, 14)
    macd_line, signal_line, macd_hist = macd(close)

    price_above_ma200 = (not np.isnan(ma200.iloc[-1])) and (last_price > ma200.iloc[-1])
    ma50_gt_ma200 = (not np.isnan(ma200.iloc[-1])) and (ma50.iloc[-1] > ma200.iloc[-1])
    rsi_ok = (not np.isnan(rsi14.iloc[-1])) and (40 <= rsi14.iloc[-1] <= 60)
    macd_bullish = (
        not (np.isnan(macd_line.iloc[-1]) or np.isnan(signal_line.iloc[-1]))
        and (macd_line.iloc[-1] > signal_line.iloc[-1])
    )
    hi_52w = hist["Close"].rolling(252, min_periods=1).max().iloc[-1]
    near_52w_high = last_price >= (hi_52w * 0.9)

    tech_score = 0
    tech_score += 10 if price_above_ma200 else 0
    tech_score += 8 if ma50_gt_ma200 else 0
    tech_score += 6 if rsi_ok else 0
    tech_score += 8 if macd_bullish else 0
    tech_score += 8 if near_52w_high else 0

    # --- Fundamental analysis ---
    pe = info.get("trailingPE", np.nan)
    pb = info.get("priceToBook", np.nan)
    roe = info.get("returnOnEquity", np.nan)
    d2e = info.get("debtToEquity", np.nan)
    rev_g = info.get("revenueGrowth", np.nan)
    pm = info.get("profitMargins", np.nan)

    pe_score = pct(pe, 5, 40, reverse=True)
    pb_score = pct(pb, 0.5, 8, reverse=True)
    roe_score = pct((roe or 0) * 100, 5, 30)
    d2e_score = pct(d2e, 0, 1.0, reverse=True)
    rev_g_score = pct((rev_g or 0) * 100, 0, 30)
    pm_score = pct((pm or 0) * 100, 0, 25)

    fundamentals_weighted = (
        12 * (pe_score / 100) +
        8  * (pb_score / 100) +
        14 * (roe_score / 100) +
        10 * (d2e_score / 100) +
        8  * (rev_g_score / 100) +
        8  * (pm_score / 100)
    )
    fundamental_score = fundamentals_weighted

    total_score = round(fundamental_score + tech_score, 2)
    signal = classify(total_score)

    return {
        "Symbol": symbol,
        "Name": info.get("shortName", symbol),
        "Price": round(last_price, 2),
        "P/E": round(pe, 2) if pe and not np.isnan(pe) else None,
        "P/B": round(pb, 2) if pb and not np.isnan(pb) else None,
        "ROE (%)": round((roe or 0) * 100, 2) if roe else None,
        "Debt/Equity": round(d2e, 2) if d2e and not np.isnan(d2e) else None,
        "Rev Growth (%)": round((rev_g or 0) * 100, 2) if rev_g else None,
        "Profit Margin (%)": round((pm or 0) * 100, 2) if pm else None,
        "RSI(14)": round(rsi14.iloc[-1], 2) if not np.isnan(rsi14.iloc[-1]) else None,
        "Tech Score (40)": round(tech_score, 2),
        "Fund Score (60)": round(fundamental_score, 2),
        "Total Score (100)": total_score,
        "Call": signal,
        "Hist": hist
    }

# --- UI ---
symbols = st.text_area(
    "Enter NSE symbols (comma-separated) – e.g. RELIANCE.NS, TCS.NS, INFY.NS",
    value="RELIANCE.NS, TCS.NS, INFY.NS"
)
symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
run_btn = st.button("Run Analysis")

if run_btn and symbols:
    results = []
    with st.spinner("Fetching & analysing..."):
        for s in symbols:
            r = analyze_symbol(s)
            if r: results.append(r)

    if not results:
        st.error("No valid data returned. Check symbols.")
    else:
        df = pd.DataFrame([
            {k: v for k, v in d.items() if k != "Hist"} for d in results
        ])
        df = df.sort_values("Total Score (100)", ascending=False)
        st.subheader("📊 Screener Results")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "screener_results.csv",
            "text/csv"
        )

        st.subheader("📈 Chart & Indicators")
        symbol_to_plot = st.selectbox("Select a symbol to view chart", df["Symbol"].tolist())
        if symbol_to_plot:
            item = next(d for d in results if d["Symbol"] == symbol_to_plot)
            hist = item["Hist"]
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"], name="Price"
            ))
            fig.update_layout(title=f"{symbol_to_plot} – Price", xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter symbols and click **Run Analysis**.")
