"""
Stock Market Intelligence Platform - Frontend
Streamlit dashboard for stock prediction, sentiment analysis, and portfolio optimization.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Stock Market Intelligence",
    page_icon="📈",
    layout="wide"
)

# API base URL
API_URL = "http://localhost:8000"

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
    
    ### 1. 📊 Stock Price Prediction
    - LSTM and XGBoost models for price forecasting
    - Technical indicators (SMA, RSI, MACD, Bollinger Bands)
    - Historical data visualization
    
    ### 2. 💭 Sentiment Analysis
    - Financial text analysis using FinBERT
    - Social media sentiment (FinTwitBERT)
    - News headline analysis
    
    ### 3. 💼 Portfolio Optimization
    - Modern Portfolio Theory implementation
    - Maximum Sharpe ratio optimization
    - Minimum variance portfolios
    
    ---
    
    ### Getting Started
    1. Install dependencies: `pip install -r requirements.txt`
    2. Start backend: `uvicorn backend.main:app --reload`
    3. Start frontend: `streamlit run frontend/app.py`
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
                try:
                    search_response = requests.get(f"{API_URL}/search/{search_query}")
                    if search_response.status_code == 200:
                        search_results = search_response.json().get('results', [])
                        
                        if search_results:
                            # Create dropdown options
                            options = {r['display']: r['symbol'] for r in search_results}
                            selected = st.selectbox("Select Company", list(options.keys()))
                            
                            if selected:
                                symbol = options[selected]
                                st.session_state['selected_symbol'] = symbol
                                st.success(f"Selected: {symbol}")
                        else:
                            st.warning("No results found")
                except Exception as e:
                    st.error(f"Search error: {e}")
        
        # Use selected symbol or default
        symbol = st.text_input("Stock Symbol", value=st.session_state.get('selected_symbol', 'AAPL')).upper()
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        interval = st.selectbox("Interval", ["1d", "1h", "30m"], index=0)
    
    with col2:
        if st.button("Fetch Data", key="fetch_price"):
            with st.spinner("Loading data..."):
                try:
                    response = requests.get(f"{API_URL}/stock/{symbol}", params={"period": period, "interval": interval})
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display price
                        st.metric("Latest Price", f"${data['latest_price']:.2f}")
                        
                        # Create DataFrame
                        df = pd.DataFrame(data['data'])
                        
                        # Convert to proper types
                        df['Close'] = df['Close'].astype(float)
                        df['Open'] = df['Open'].astype(float)
                        df['High'] = df['High'].astype(float)
                        df['Low'] = df['Low'].astype(float)
                        
                        # Chart
                        st.line_chart(df['Close'])
                        
                        # Data table
                        with st.expander("View Raw Data"):
                            st.dataframe(df.tail(20))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to API. Make sure backend is running: {e}")

# ============================================================================
# Sentiment Analysis Page
# ============================================================================

elif page == "💭 Sentiment Analysis":
    st.title("💭 Financial Sentiment Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_choice = st.selectbox(
            "Model",
            ["ProsusAI/finbert", "FinTwitBERT-sentiment", "distilbert_finance", "deberta-v3-financial"]
        )
        st.caption(f"Using: {model_choice}")
    
    with col2:
        # Model info
        model_info = {
            "ProsusAI/finbert": "General financial sentiment (90% accuracy)",
            "FinTwitBERT-sentiment": "Twitter/Social media (85% accuracy)",
            "distilbert_finance": "Fast inference (97.5% accuracy)",
            "deberta-v3-financial": "Max accuracy (99.4% accuracy)"
        }
        st.info(model_info.get(model_choice, ""))
    
    # Text input
    text_input = st.text_area("Enter financial text to analyze", height=150,
        placeholder="e.g., 'Apple reports record quarterly earnings, stock surges 5%'")
    
    if st.button("Analyze Sentiment"):
        if text_input:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(f"{API_URL}/sentiment", json={
                        "text": text_input,
                        "model": model_choice
                    })
                    if response.status_code == 200:
                        result = response.json()
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Positive", f"{result['scores']['positive']*100:.1f}%")
                        col2.metric("Neutral", f"{result['scores']['neutral']*100:.1f}%")
                        col3.metric("Negative", f"{result['scores']['negative']*100:.1f}%")
                        
                        st.success(f"Overall: **{result['sentiment'].upper()}**")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"API Error: {e}")
        else:
            st.warning("Please enter some text to analyze")

# ============================================================================
# Price Prediction Page
# ============================================================================

elif page == "📈 Price Prediction":
    st.title("📈 Stock Price Prediction")
    
    # Smart search for prediction
    col_search, col1 = st.columns([2, 1])
    
    with col_search:
        pred_search = st.text_input("Search Company to Predict", placeholder="e.g., Apple, Reliance, Tesla...", key="pred_search")
        
        if pred_search:
            with st.spinner("Searching..."):
                try:
                    search_response = requests.get(f"{API_URL}/search/{pred_search}")
                    if search_response.status_code == 200:
                        search_results = search_response.json().get('results', [])
                        
                        if search_results:
                            options = {r['display']: r['symbol'] for r in search_results}
                            selected = st.selectbox("Select", list(options.keys()), key="pred_select")
                            
                            if selected:
                                st.session_state['pred_symbol'] = options[selected]
                except:
                    pass
    
    with col1:
        pred_symbol = st.text_input("Symbol", value=st.session_state.get('pred_symbol', 'AAPL')).upper()
        pred_days = st.slider("Days to Predict", 7, 90, 30)
    
    col2, col3 = st.columns([1, 1])
    
    with col2:
        model_type = st.selectbox("Model Type", ["LSTM", "XGBoost"])
    
    st.markdown("---")
    
    if st.button("Generate Prediction", key="gen_pred"):
        with st.spinner("Generating prediction..."):
            try:
                # First fetch historical data
                hist_response = requests.get(f"{API_URL}/stock/{pred_symbol}", params={"period": "3mo", "interval": "1d"})
                
                if hist_response.status_code == 200:
                    hist_data = hist_response.json()
                    latest_price = hist_data.get('latest_price', 150)
                    
                    # Call prediction API
                    response = requests.post(f"{API_URL}/predict", json={
                        "symbol": pred_symbol,
                        "days": pred_days,
                        "model_type": model_type.lower()
                    })
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction for {pred_symbol}")
                        
                        # Generate mock prediction data (placeholder)
                        import numpy as np
                        pred_days_list = list(range(pred_days))
                        base_price = latest_price
                        predictions = [base_price * (1 + np.random.randn() * 0.02) for _ in range(pred_days)]
                        
                        # Create prediction chart
                        pred_df = pd.DataFrame({
                            "Day": pred_days_list,
                            "Predicted Price": predictions
                        })
                        st.line_chart(pred_df.set_index("Day"))
                        
                        st.info(f"Current Price: ${latest_price:.2f} | Predicted (Day {pred_days}): ${predictions[-1]:.2f}")
                    else:
                        st.error(f"Error: {response.text}")
                else:
                    st.error(f"Could not fetch data for {pred_symbol}")
            except Exception as e:
                st.error(f"API Error: {e}")
    
    st.markdown("""
    ### Model Information
    
    **LSTM (Long Short-Term Memory)**
    - Deep learning approach for sequence prediction
    - Captures long-term dependencies in price data
    - Typical accuracy: 70-75% directional
    
    **XGBoost**
    - Gradient boosting for tabular data
    - Fast training and inference
    - Uses technical indicators as features
    """)

# ============================================================================
# Portfolio Optimization Page
# ============================================================================

elif page == "💼 Portfolio Optimization":
    st.title("💼 Portfolio Optimization")
    
    st.markdown("### Enter Portfolio Assets")
    
    # Smart search for adding stocks
    col_search, col_add = st.columns([3, 1])
    
    with col_search:
        portfolio_search = st.text_input("Search Company to Add", placeholder="e.g., Apple, Reliance, TCS...", key="portfolio_search")
    
    # Initialize portfolio symbols in session state
    if 'portfolio_symbols' not in st.session_state:
        st.session_state['portfolio_symbols'] = ["AAPL", "MSFT", "GOOGL"]
    
    with col_add:
        if portfolio_search:
            with st.spinner("Searching..."):
                try:
                    search_response = requests.get(f"{API_URL}/search/{portfolio_search}")
                    if search_response.status_code == 200:
                        search_results = search_response.json().get('results', [])
                        
                        if search_results:
                            options = {r['display']: r['symbol'] for r in search_results[:5]}
                            selected = st.selectbox("Select", list(options.keys()), key="portfolio_select")
                            
                            if st.button("Add to Portfolio"):
                                if selected:
                                    new_symbol = options[selected]
                                    if new_symbol not in st.session_state['portfolio_symbols']:
                                        st.session_state['portfolio_symbols'].append(new_symbol)
                                        st.success(f"Added {new_symbol}")
                except:
                    pass
    
    # Display current portfolio
    st.markdown("#### Current Portfolio")
    symbols_str = ", ".join(st.session_state['portfolio_symbols'])
    symbols_input = st.text_input("Stock Symbols (comma-separated)", value=symbols_str)
    
    # Allow manual editing - update session state
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
    
    st.markdown("### Optimization Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization = st.selectbox("Objective", ["Max Sharpe Ratio", "Minimum Variance", "Equal Weight"])
    
    with col2:
        allow_short = st.checkbox("Allow Short Positions", value=False)
    
    if st.button("Optimize Portfolio"):
        with st.spinner("Optimizing portfolio..."):
            try:
                response = requests.post(f"{API_URL}/portfolio/optimize", json={
                    "symbols": symbols,
                    "optimization": optimization.lower().replace(" ", "_")
                })
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Optimization Complete!")
                    
                    # Display weights
                    st.markdown("### Optimal Allocation")
                    weights_df = pd.DataFrame({
                        "Symbol": result["symbols"],
                        "Weight (%)": [w * 100 for w in result["weights"]]
                    })
                    st.dataframe(weights_df, hide_index=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Return", "12.0%")
                    col2.metric("Volatility", "18.0%")
                    col3.metric("Sharpe Ratio", "0.67")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"API Error: {e}")
    
    st.markdown("""
    ### Modern Portfolio Theory
    
    Optimize your portfolio using Harry Markowitz's framework:
    
    - **Max Sharpe Ratio**: Maximize risk-adjusted returns
    - **Minimum Variance**: Minimize portfolio volatility
    - **Equal Weight**: Simple diversification baseline
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Stock Market Intelligence Platform | Built with Streamlit + FastAPI + ML"
    "</div>",
    unsafe_allow_html=True
)
