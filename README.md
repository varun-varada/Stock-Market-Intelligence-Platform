# Stock Market Intelligence Platform

A comprehensive ML-powered stock market analysis platform with price prediction, sentiment analysis, and portfolio optimization.

## 📋 Overview

This platform integrates three core modules:

| Module | Description | Technologies |
|--------|-------------|--------------|
| **Price Prediction** | LSTM & XGBoost models for stock forecasting | TensorFlow, XGBoost, Scikit-learn |
| **Sentiment Analysis** | Financial text analysis using FinBERT | Transformers, HuggingFace |
| **Portfolio Optimization** | Modern Portfolio Theory implementation | CVXPY, SciPy |

## 🏗️ Project Structure

```
ml_project/
├── backend/
│   └── main.py           # FastAPI server
├── frontend/
│   └── app.py            # Streamlit dashboard
├── src/
│   ├── data_acquisition.py
│   ├── sentiment_analysis.py
│   ├── price_prediction.py
│   └── portfolio_optimization.py
├── data/                 # Data storage
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── docs/                 # Documentation
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ml_project
pip install -r requirements.txt
```

**Note:** If TA-Lib fails to install, you can skip it - it's optional for basic functionality.

### 2. Start Backend

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Start Frontend (in a new terminal)

```bash
cd frontend
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Available Models

### Sentiment Analysis

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| ProsusAI/finbert | ~90% | Medium | General financial text |
| FinTwitBERT | ~85% | Medium | Twitter/social media |
| DistilBERT Finance | 97.5% | Fast | High throughput |
| DeBERTa Financial | 99.4% | Fast | Maximum accuracy |

### Price Prediction

- **LSTM**: Deep learning for sequence prediction
- **XGBoost**: Gradient boosting with technical indicators

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/stock/{symbol}` | GET | Historical stock data |
| `/stock/{symbol}/info` | GET | Stock metadata |
| `/sentiment` | POST | Analyze text sentiment |
| `/portfolio/optimize` | POST | Optimize portfolio |
| `/predict` | POST | Predict stock prices |
| `/health` | GET | Health check |

## 💻 Usage Examples

### Fetch Stock Data

```python
import requests

response = requests.get("http://localhost:8000/stock/AAPL", params={
    "period": "1y",
    "interval": "1d"
})
data = response.json()
```

### Analyze Sentiment

```python
import requests

response = requests.post("http://localhost:8000/sentiment", json={
    "text": "Apple reports record earnings, stock surges 5%",
    "model": "prosusai/finbert"
})
result = response.json()
# {'sentiment': 'positive', 'scores': {...}}
```

### Optimize Portfolio

```python
import requests

response = requests.post("http://localhost:8000/portfolio/optimize", json={
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "optimization": "max_sharpe"
})
result = response.json()
```

## 🔧 Configuration

### Backend (backend/main.py)

- API port: 8000
- CORS enabled for Streamlit

### Frontend (frontend/app.py)

- Streamlit port: 8501
- API URL: http://localhost:8000

### Models

- Default sentiment model: ProsusAI/finbert
- Default prediction: LSTM
- Sequence length: 60 days

## 📦 Dependencies

**Core:**
- streamlit, fastapi, uvicorn
- yfinance, pandas, numpy

**ML:**
- tensorflow, xgboost, scikit-learn
- transformers, torch

**Optimization:**
- scipy, cvxpy

## ⚠️ Disclaimer

This platform is for educational purposes only. Stock predictions and portfolio optimizations are based on historical data and do not guarantee future performance. Always consult a financial advisor before making investment decisions.

## 📚 References

- [FinBERT - Financial Sentiment Analysis](https://huggingface.co/ProsusAI/finbert)
- [Modern Portfolio Theory - Markowitz (1952)](https://www.jstor.org/stable/2977734)
- [LSTM for Stock Prediction](https://arxiv.org/abs/1709.08491)

---

**Built with ❤️ using Streamlit, FastAPI, TensorFlow, and Transformers**
