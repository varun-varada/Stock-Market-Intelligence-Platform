"""
Stock Market Intelligence Platform - Backend API
FastAPI server for price prediction, sentiment analysis, and portfolio optimization.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI(
    title="Stock Market Intelligence API",
    description="Backend API for stock prediction, sentiment, and portfolio optimization",
    version="1.0.0"
)

# ============================================================================
# Data Models
# ============================================================================

class StockQuery(BaseModel):
    symbol: str
    period: str = "1y"  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    interval: str = "1d" # 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo

class SentimentQuery(BaseModel):
    text: str
    model: str = "prosusai/finbert"  # Model selection

class PortfolioInput(BaseModel):
    symbols: List[str]
    weights: Optional[List[float]] = None
    optimization: str = "max_sharpe"  # max_sharpe, min_variance, equal_weight

class PredictionInput(BaseModel):
    symbol: str
    days: int = 30
    model_type: str = "lstm"  # lstm, xgboost

# ============================================================================
# Data Acquisition Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "Stock Market Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "stock_data": "GET /stock/{symbol} - Get historical stock data",
            "sentiment": "POST /sentiment - Analyze text sentiment",
            "portfolio": "POST /portfolio/optimize - Optimize portfolio",
            "predict": "POST /predict - Predict stock prices",
        }
    }

@app.get("/stock/{symbol}")
async def get_stock_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d"
):
    """Fetch historical stock data using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Add technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Handle NaN values for JSON serialization
        df = df.fillna(0)
        
        return {
            "symbol": symbol,
            "data": df.tail(100).to_dict(orient="records"),
            "latest_price": float(df['Close'].iloc[-1]),
            "period": period,
            "interval": interval
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str):
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
            "dividend_yield": info.get("dividendYield", "N/A"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Sentiment Analysis Endpoints
# ============================================================================

@app.post("/sentiment")
async def analyze_sentiment(query: SentimentQuery):
    """Analyze sentiment of financial text using FinBERT."""
    try:
        from transformers import pipeline
        
        # Map model names to HuggingFace model IDs
        model_map = {
            "ProsusAI/finbert": "ProsusAI/finbert",
            "FinTwitBERT-sentiment": "Finnish-NLP/FinTwitBERT-sentiment",
            "distilbert_finance": "dslim/bert-base-NER",  # fallback
            "deberta-v3-financial": "microsoft/deberta-v3-base"  # fallback
        }
        
        model_id = model_map.get(query.model, "ProsusAI/finbert")
        
        # Use sentiment-analysis pipeline
        classifier = pipeline("sentiment-analysis", model=model_id, truncation=True, max_length=512)
        
        result = classifier(query.text)[0]
        
        # Convert to our format
        label = result['label'].lower()
        score = result['score']
        
        if label == "positive":
            scores = {"positive": score, "negative": 0.0, "neutral": 1.0 - score}
        elif label == "negative":
            scores = {"positive": 0.0, "negative": score, "neutral": 1.0 - score}
        else:
            scores = {"positive": 0.0, "negative": 0.0, "neutral": score}
        
        return {
            "text": query.text,
            "model": query.model,
            "sentiment": label,
            "scores": scores
        }
    except Exception as e:
        # Fallback to placeholder if model fails
        return {
            "text": query.text,
            "model": query.model,
            "sentiment": "neutral",
            "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            "error": str(e)
        }

# ============================================================================
# Portfolio Optimization Endpoints
# ============================================================================

@app.post("/portfolio/optimize")
async def optimize_portfolio(input_data: PortfolioInput):
    """Optimize portfolio based on Modern Portfolio Theory."""
    import numpy as np
    
    symbols = input_data.symbols
    
    try:
        # Fetch historical data
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
            weights = [1.0 / len(symbols)] * len(symbols)
            return {
                "symbols": symbols,
                "weights": weights,
                "expected_return": 12.0,
                "volatility": 18.0,
                "sharpe_ratio": 0.67,
                "message": "Not enough data"
            }
        
        # Stack returns
        returns_matrix = np.array(returns_data)
        mean_returns = np.mean(returns_matrix, axis=1)
        cov_matrix = np.cov(returns_matrix)
        
        n = len(valid_symbols)
        
        if input_data.optimization == "max_sharpe":
            # Simple risk-parity style optimization (approximate max sharpe)
            inv_vol = 1.0 / (np.sqrt(np.diag(cov_matrix)) + 1e-10)
            weights = inv_vol / np.sum(inv_vol)
        elif input_data.optimization == "min_variance":
            # Minimum variance (inverse of variance)
            inv_var = 1.0 / (np.diag(cov_matrix) + 1e-10)
            weights = inv_var / np.sum(inv_var)
        else:  # equal_weight
            weights = np.array([1.0/n] * n)
        
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        expected_return = float(mean_returns @ weights) * 252  # Annualized
        volatility = float(np.sqrt(weights @ cov_matrix @ weights)) * np.sqrt(252)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        return {
            "symbols": valid_symbols,
            "weights": [round(float(w), 4) for w in weights],
            "expected_return": round(expected_return * 100, 2),
            "volatility": round(volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "optimization": input_data.optimization
        }
        
    except Exception as e:
        weights = [1.0 / len(symbols)] * len(symbols)
        return {
            "symbols": symbols,
            "weights": weights,
            "expected_return": 12.0,
            "volatility": 18.0,
            "sharpe_ratio": 0.67,
            "message": f"Using equal weights ({str(e)[:40]})"
        }

# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict")
async def predict_stock(input_data: PredictionInput):
    """Predict future stock prices."""
    # Placeholder - integrate LSTM/XGBoost model here
    return {
        "symbol": input_data.symbol,
        "days": input_data.days,
        "model_type": input_data.model_type,
        "predictions": [],
        "message": "Model not yet trained. Train model first."
    }

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/search/{query}")
async def search_stocks(query: str):
    """Search for stocks by company name using Yahoo Finance."""
    try:
        search_results = yf.Search(query).quotes
        
        results = []
        for r in search_results:
            if r.get('quoteType') in ['EQUITY', 'ETF']:
                exchange = r.get('exchDisp', r.get('exchange', 'Unknown'))
                symbol = r.get('symbol', '')
                name = r.get('longname') or r.get('shortname') or symbol
                
                # Format display string
                display = f"{name} ({symbol} - {exchange})"
                
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": exchange,
                    "type": r.get('quoteType', 'Unknown'),
                    "display": display
                })
        
        return {
            "query": query,
            "results": results[:10]
        }
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
