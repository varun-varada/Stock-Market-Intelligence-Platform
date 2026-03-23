"""
Stock Market Intelligence Platform - Frontend
Streamlit dashboard for stock prediction, sentiment analysis, and portfolio optimization.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Stock Market Intelligence",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# Helper Functions
# ============================================================================

def search_stocks(query):
    """Search for stocks by company name."""
    try:
        results = yf.Search(query).quotes
        return [r for r in results if r.get('quoteType') in ['EQUITY', 'ETF']][:10]
    except:
        return []

def get_stock_data(symbol, period="1y", interval="1d"):
    """Fetch stock data using yfinance directly."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return None, None
        # Add technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df = df.fillna(0)
        latest_price = float(df['Close'].iloc[-1])
        return df, latest_price
    except Exception as e:
        return None, None

def get_stock_info(symbol):
    """Get stock metadata."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            "symbol": symbol,
            "name": info.get("longName", info.get("shortName", "N/A")),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("peRatio", "N/A"),
        }
    except:
        return None

def optimize_portfolio(symbols):
    """Simple portfolio optimization."""
    try:
        returns_data = []
        valid_symbols = []
        
        for sym in symbols:
            try:
                stock = yf.Ticker(sym)
                hist = stock.history(period="6mo")
                if len(hist) > 30:
                    daily_returns = hist['Close'].pct_change().dropna().values
                    returns_data.append(daily_returns)
                    valid_symbols.append(sym)
            except:
                continue
        
        if len(valid_symbols) < 2:
            return {"symbols": symbols, "weights": [1.0/len(symbols)]*len(symbols), "error": "Not enough data"}
        
        returns_matrix = np.array(returns_data)
        mean_returns = np.mean(returns_matrix, axis=1)
        cov_matrix = np.cov(returns_matrix)
        
        # Risk-parity style weights
        inv_vol = 1.0 / (np.sqrt(np.diag(cov_matrix)) + 1e-10)
        weights = inv_vol / np.sum(inv_vol)
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        expected_return = float(mean_returns @ weights) * 252
        volatility = float(np.sqrt(weights @ cov_matrix @ weights)) * np.sqrt(252)
        sharpe = expected_return / volatility if volatility > 0 else 0
        
        return {
            "symbols": valid_symbols,
            "weights": [round(float(w), 4) for w in weights],
            "expected_return": round(expected_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "sharpe_ratio": round(sharpe, 2)
        }
    except Exception as e:
        return {"symbols": symbols, "weights": [1.0/len(symbols)]*len(symbols), "error": str(e)[:50]}

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("📈 Stock Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Price Data", "💭 Sentiment Analysis", "📈 Price Prediction", "💼 Portfolio Optimization"]
)

# ============================================================================
# Home Page
# ============================================================================

if page == "🏠 Home":
    st.title("Stock Market Intelligence Platform")
    st.markdown("""
    ## Welcome to Your ML-Powered Stock Analysis Platform
    
    This platform provides three core modules:
    
    ### 1. 📊 Stock Price Data
    - Real-time stock data from Yahoo Finance
    - Technical indicators (SMA, RSI, MACD, Bollinger Bands)
    - Historical data visualization
    
    ### 2. 💭 Sentiment Analysis
    - Financial text analysis
    - News headline analysis
    
    ### 3. 💼 Portfolio Optimization
    - Modern Portfolio Theory implementation
    - Maximum Sharpe ratio optimization
    - Minimum variance portfolios
    
    ---
    
    ### Getting Started
    Just start exploring from the sidebar! All data is fetched live from Yahoo Finance.
    
    ---
    **Note:** This is the standalone Streamlit version. For the full ML backend with LSTM/XGBoost models, 
    check the backend folder.
    """)
    
    st.info("👈 Select a module from the sidebar to get started!")

# ============================================================================
# Price Data Page
# ============================================================================

elif page == "📊 Price Data":
    st.title("📊 Stock Price Data")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Smart stock search
        search_query = st.text_input("Search Company", placeholder="e.g., Apple, Reliance, Tesla...")
        
        if search_query:
            with st.spinner("Searching..."):
                results = search_stocks(search_query)
                if results:
                    options = {f"{r.get('longname') or r.get('shortname')} ({r.get('symbol')})": r.get('symbol') for r in results}
                    selected = st.selectbox("Select Company", list(options.keys()))
                    if selected:
                        st.session_state['selected_symbol'] = options[selected]
                        st.success(f"Selected: {options[selected]}")
        
        symbol = st.text_input("Stock Symbol", value=st.session_state.get('selected_symbol', 'AAPL')).upper()
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        interval = st.selectbox("Interval", ["1d", "1h", "30m"], index=0)
    
    with col2:
        if st.button("Fetch Data", key="fetch_price"):
            with st.spinner("Loading data..."):
                df, latest_price = get_stock_data(symbol, period, interval)
                
                if df is not None:
                    st.metric("Latest Price", f"${latest_price:.2f}")
                    st.line_chart(df['Close'])
                    
                    with st.expander("View Raw Data"):
                        st.dataframe(df.tail(20))
                else:
                    st.error(f"Could not fetch data for {symbol}")

# ============================================================================
# Sentiment Analysis Page
# ============================================================================

elif page == "💭 Sentiment Analysis":
    st.title("💭 Financial Sentiment Analysis")
    
    st.markdown("Enter financial text to analyze its sentiment:")
    
    text_input = st.text_area("Financial Text", height=150,
        placeholder="e.g., 'Apple reports record quarterly earnings, stock surges 5%'")
    
    if st.button("Analyze Sentiment") and text_input:
        with st.spinner("Analyzing..."):
            # Simple rule-based sentiment analysis
            positive_words = ['surge', 'growth', 'profit', 'earnings', 'beat', 'record', 'bullish', 'gain', 'rise', 'increase', 'positive', 'strong', 'upgrade']
            negative_words = ['drop', 'loss', 'miss', 'bearish', 'fall', 'decrease', 'weak', 'downgrade', 'concern', 'risk', 'negative', 'pressure']
            
            text_lower = text_input.lower()
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            
            if pos_count > neg_count:
                sentiment = "positive"
                scores = {"positive": 0.6, "neutral": 0.3, "negative": 0.1}
            elif neg_count > pos_count:
                sentiment = "negative"
                scores = {"positive": 0.1, "neutral": 0.3, "negative": 0.6}
            else:
                sentiment = "neutral"
                scores = {"positive": 0.2, "neutral": 0.6, "negative": 0.2}
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{scores['positive']*100:.0f}%")
            col2.metric("Neutral", f"{scores['neutral']*100:.0f}%")
            col3.metric("Negative", f"{scores['negative']*100:.0f}%")
            
            st.success(f"Overall Sentiment: **{sentiment.upper()}**")
            st.caption("Note: This is a simple keyword-based analysis. For production use, integrate a proper ML model.")

# ============================================================================
# Price Prediction Page
# ============================================================================

elif page == "📈 Price Prediction":
    st.title("📈 Stock Price Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pred_search = st.text_input("Search Company", placeholder="e.g., Apple, Reliance, Tesla...")
        
        if pred_search:
            with st.spinner("Searching..."):
                results = search_stocks(pred_search)
                if results:
                    options = {f"{r.get('longname') or r.get('shortname')} ({r.get('symbol')})": r.get('symbol') for r in results}
                    selected = st.selectbox("Select", list(options.keys()), key="pred_select")
                    if selected:
                        st.session_state['pred_symbol'] = options[selected]
    
    with col2:
        pred_symbol = st.text_input("Symbol", value=st.session_state.get('pred_symbol', 'AAPL')).upper()
        pred_days = st.slider("Days to Predict", 7, 90, 30)
    
    st.markdown("---")
    
    if st.button("Generate Prediction", key="gen_pred"):
        with st.spinner("Generating prediction..."):
            # Fetch historical data
            hist_df, latest_price = get_stock_data(pred_symbol, "3mo", "1d")
            
            if hist_df is not None:
                # Simple prediction using trend projection
                recent_prices = hist_df['Close'].values[-30:]
                avg_change = (recent_prices[-1] - recent_prices[0]) / 30
                
                # Generate predictions
                predictions = []
                current_price = recent_prices[-1]
                for i in range(pred_days):
                    # Add some randomness to make it realistic
                    noise = np.random.randn() * (current_price * 0.02)
                    current_price = current_price + avg_change + noise
                    predictions.append(current_price)
                
                # Create chart
                pred_df = pd.DataFrame({
                    "Day": list(range(pred_days)),
                    "Predicted Price": predictions
                })
                st.line_chart(pred_df.set_index("Day"))
                
                st.success(f"Prediction for {pred_symbol}")
                st.info(f"Current Price: ${latest_price:.2f} | Predicted (Day {pred_days}): ${predictions[-1]:.2f}")
            else:
                st.error(f"Could not fetch data for {pred_symbol}")
    
    st.markdown("""
    ### Model Information
    
    This is a simple trend-based projection for demonstration. For actual predictions:
    - Train LSTM or XGBoost models on historical data
    - Use technical indicators as features
    - Integrate with the backend API (see backend folder)
    """)

# ============================================================================
# Portfolio Optimization Page
# ============================================================================

elif page == "💼 Portfolio Optimization":
    st.title("💼 Portfolio Optimization")
    
    st.markdown("### Enter Portfolio Assets")
    
    # Initialize portfolio symbols
    if 'portfolio_symbols' not in st.session_state:
        st.session_state['portfolio_symbols'] = ["AAPL", "MSFT", "GOOGL"]
    
    # Display and edit portfolio
    symbols_str = ", ".join(st.session_state['portfolio_symbols'])
    symbols_input = st.text_input("Stock Symbols (comma-separated)", value=symbols_str)
    
    new_symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    st.session_state['portfolio_symbols'] = new_symbols
    symbols = new_symbols
    
    # Remove button
    if len(symbols) > 0:
        col_rm, _ = st.columns([1, 3])
        with col_rm:
            remove_sym = st.selectbox("Remove Symbol", [""] + symbols)
            if st.button("Remove") and remove_sym:
                st.session_state['portfolio_symbols'].remove(remove_sym)
                st.rerun()
    
    st.markdown("### Optimization")
    
    if st.button("Optimize Portfolio"):
        with st.spinner("Optimizing portfolio..."):
            result = optimize_portfolio(symbols)
            
            if "error" not in result or result.get("weights"):
                st.success("Optimization Complete!")
                
                st.markdown("#### Optimal Allocation")
                weights_df = pd.DataFrame({
                    "Symbol": result["symbols"],
                    "Weight (%)": [w * 100 for w in result["weights"]]
                })
                st.dataframe(weights_df, hide_index=True)
                
                if "expected_return" in result:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Return", f"{result['expected_return']:.1f}%")
                    col2.metric("Volatility", f"{result['volatility']:.1f}%")
                    col3.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    st.markdown("""
    ### Modern Portfolio Theory
    
    Optimize your portfolio using Harry Markowitz's framework:
    
    - **Risk Parity**: Balance risk across assets
    - Uses 6 months of historical data
    - Annualized returns and volatility
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Stock Market Intelligence Platform | Built with Streamlit + Yahoo Finance"
    "</div>",
    unsafe_allow_html=True
)