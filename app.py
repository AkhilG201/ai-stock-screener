import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(page_title="AI Stock Screener", layout="wide")

st.title("📈 AI-Powered Stock Screener for NSE")

symbol = st.text_input("🔍 Enter NSE stock symbol (e.g., RELIANCE.NS, INFY.NS)", value="RELIANCE.NS")

if symbol:
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")
        info = stock.info

        st.subheader(f"Company: {info.get('shortName', 'N/A')}")
        sector = info.get('sector', 'N/A')
        market_cap = f"₹{info.get('marketCap', 0):,}" if info.get('marketCap') else 'N/A'
        pe_ratio = info.get('trailingPE', 'N/A')
        high_52w = info.get('fiftyTwoWeekHigh', 'N/A')
        low_52w = info.get('fiftyTwoWeekLow', 'N/A')

        st.markdown(f"""
        - **Sector**: {sector}
        - **Market Cap**: {market_cap}
        - **PE Ratio**: {pe_ratio}
        - **52W High**: ₹{high_52w}
        - **52W Low**: ₹{low_52w}
        """)

        st.subheader("📊 Live Chart (1Y)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Close Price"))
        fig.update_layout(title=f"{symbol} Stock Price", xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

        df_stats = pd.DataFrame.from_dict(info, orient='index', columns=['Value']).dropna()
        st.subheader("📑 Full Financial Metrics")
        st.dataframe(df_stats)

        st.download_button("📥 Download Financials to CSV", df_stats.to_csv().encode('utf-8'), f"{symbol}_metrics.csv", "text/csv")

        st.subheader("💡 AI Recommendation (Demo)")
        pe = info.get("trailingPE", None)
        rating = "Hold"
        if pe:
            if pe < 15:
                rating = "Buy"
            elif pe > 30:
                rating = "Sell"
        st.success(f"✅ **Recommendation**: {rating} (based on PE Ratio)")

    except Exception as e:
        st.error(f"Error fetching data: {e}")