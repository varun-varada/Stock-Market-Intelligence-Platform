"""
Sentiment Analysis Module
Uses HuggingFace transformers for financial sentiment analysis.
"""

from typing import Dict, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


# Model options
MODELS = {
    "prosusai/finbert": {
        "description": "General financial sentiment",
        "accuracy": "~90%",
        "params": "110M"
    },
    "StephanAkkerman/FinTwitBERT-sentiment": {
        "description": "Twitter/Social media",
        "accuracy": "~85%",
        "params": "110M"
    },
    "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis": {
        "description": "Max accuracy",
        "accuracy": "99.4%",
        "params": "44M"
    },
    "AdityaAI9/distilbert_finance_sentiment_analysis": {
        "description": "Fast inference",
        "accuracy": "97.5%",
        "params": "66M"
    }
}


class SentimentAnalyzer:
    """Financial sentiment analyzer using HuggingFace models."""
    
    def __init__(self, model_name: str = "prosusai/finbert"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment and scores
        """
        # Handle long texts by truncation
        max_length = 512
        text = text[:max_length]
        
        result = self.pipeline(text)[0]
        
        return {
            "text": text,
            "sentiment": result["label"],
            "score": result["score"]
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def get_scores(self, text: str) -> Dict[str, float]:
        """
        Get normalized sentiment scores (positive, negative, neutral).
        
        Args:
            text: Input text
            
        Returns:
            Dict with normalized scores
        """
        result = self.analyze(text)
        label = result["sentiment"].lower()
        score = result["score"]
        
        # Normalize to three classes
        if label == "positive":
            return {"positive": score, "neutral": (1 - score) / 2, "negative": (1 - score) / 2}
        elif label == "negative":
            return {"negative": score, "neutral": (1 - score) / 2, "positive": (1 - score) / 2}
        else:
            return {"neutral": score, "positive": (1 - score) / 2, "negative": (1 - score) / 2}


def analyze_financial_news(news_headlines: List[str], model: str = "prosusai/finbert") -> List[Dict]:
    """
    Analyze sentiment of financial news headlines.
    
    Args:
        news_headlines: List of news headlines
        model: Model to use
        
    Returns:
        List of sentiment results
    """
    analyzer = SentimentAnalyzer(model)
    return analyzer.analyze_batch(news_headlines)


# Example usage
if __name__ == "__main__":
    # Test sentiment analysis
    analyzer = SentimentAnalyzer("prosusai/finbert")
    
    test_texts = [
        "Apple reports record earnings, stock surges 5%",
        "Market faces uncertainty amid rate hike fears",
        "Company announces new product launch"
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} ({result['score']:.3f})")
        print()
