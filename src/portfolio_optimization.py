"""
Portfolio Optimization Module
Implements Modern Portfolio Theory using CVXPY.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import cvxpy as cp


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory."""
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize optimizer with historical returns.
        
        Args:
            returns: DataFrame of asset returns (columns = assets)
        """
        self.returns = returns
        self.expected_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
    
    def optimize(
        self,
        objective: str = "max_sharpe",
        risk_free_rate: float = 0.05,
        allow_short: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Optimize portfolio weights.
        
        Args:
            objective: Optimization objective
                      "max_sharpe" - Maximum Sharpe ratio
                      "min_variance" - Minimum variance
                      "max_return" - Maximum expected return
            risk_free_rate: Risk-free rate for Sharpe ratio
            allow_short: Allow short positions
            
        Returns:
            Optimal weights and portfolio metrics
        """
        # Define weight variable
        if allow_short:
            w = cp.Variable(self.n_assets)
        else:
            w = cp.Variable(self.n_assets, nonneg=True)
        
        # Expected return
        portfolio_return = self.expected_returns.values @ w
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
        portfolio_std = cp.sqrt(portfolio_variance)
        
        # Constraints
        constraints = [cp.sum(w) == 1]
        
        # Objective based on strategy
        if objective == "max_sharpe":
            # Maximize Sharpe ratio (minimize negative)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_std
            objective = cp.Maximize(sharpe)
            
        elif objective == "min_variance":
            # Minimize variance
            objective = cp.Minimize(portfolio_variance)
            
        elif objective == "max_return":
            # Maximize return for given variance
            target_vol = 0.15  # 15% target
            constraints.append(portfolio_std <= target_vol)
            objective = cp.Maximize(portfolio_return)
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed: {problem.status}")
        
        weights = w.value
        
        # Calculate metrics
        metrics = {
            "expected_return": float(portfolio_return.value),
            "volatility": float(portfolio_std.value),
            "sharpe_ratio": float((portfolio_return.value - risk_free_rate) / portfolio_std.value),
            "variance": float(portfolio_variance.value)
        }
        
        return weights, metrics
    
    def efficient_frontier(
        self,
        n_points: int = 50,
        allow_short: bool = False
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier.
        
        Args:
            n_points: Number of points on the frontier
            allow_short: Allow short positions
            
        Returns:
            DataFrame with returns, volatilities, and weights
        """
        # Get return range
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        results = []
        
        for target in target_returns:
            try:
                w = cp.Variable(self.n_assets)
                if not allow_short:
                    w = cp.Variable(self.n_assets, nonneg=True)
                
                portfolio_return = self.expected_returns.values @ w
                portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
                portfolio_std = cp.sqrt(portfolio_variance)
                
                constraints = [
                    cp.sum(w) == 1,
                    portfolio_return >= target
                ]
                
                if not allow_short:
                    constraints.append(w >= 0)
                
                objective = cp.Minimize(portfolio_variance)
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS)
                
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    results.append({
                        "return": target,
                        "volatility": portfolio_std.value,
                        "weights": w.value
                    })
            except:
                continue
        
        return pd.DataFrame(results)
    
    def risk_parity(
        self,
        risk_target: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute risk parity weights.
        
        Args:
            risk_target: Target portfolio risk (optional)
            
        Returns:
            Weights and metrics
        """
        # Risk parity: each asset contributes equally to portfolio risk
        cov = self.cov_matrix.values
        n = self.n_assets
        
        # Initial guess
        w = cp.Variable(n, nonneg=True)
        
        # Risk contribution
        sigma = cp.sqrt(cp.quad_form(w, cov))
        risk_contrib = w * (cov @ w) / sigma
        
        # Target: equal risk contribution
        target_risk = cp.Variable(nonneg=True)
        
        constraints = [
            cp.sum(w) == 1,
            risk_contrib == target_risk
        ]
        
        objective = cp.Minimize(0)  # Any feasible solution
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        weights = w.value / w.value.sum()  # Normalize
        
        metrics = {
            "expected_return": float(self.expected_returns.values @ weights),
            "volatility": float(np.sqrt(weights @ cov @ weights)),
            "sharpe_ratio": float((self.expected_returns.values @ weights - 0.05) / 
                                   np.sqrt(weights @ cov @ weights))
        }
        
        return weights, metrics


def optimize_portfolio(
    symbols: List[str],
    historical_returns: pd.DataFrame,
    objective: str = "max_sharpe",
    allow_short: bool = False
) -> dict:
    """
    Convenience function for portfolio optimization.
    
    Args:
        symbols: List of asset symbols
        historical_returns: DataFrame of historical returns
        objective: Optimization objective
        allow_short: Allow short positions
        
    Returns:
        Dict with weights and metrics
    """
    optimizer = PortfolioOptimizer(historical_returns)
    weights, metrics = optimizer.optimize(objective, allow_short=allow_short)
    
    return {
        "symbols": symbols,
        "weights": weights.tolist(),
        "metrics": metrics
    }


# Example usage
if __name__ == "__main__":
    # Example: Create dummy returns
    np.random.seed(42)
    n_days = 252
    returns = pd.DataFrame({
        "AAPL": np.random.randn(n_days) * 0.02,
        "MSFT": np.random.randn(n_days) * 0.018,
        "GOOGL": np.random.randn(n_days) * 0.022
    })
    
    optimizer = PortfolioOptimizer(returns)
    weights, metrics = optimizer.optimize("max_sharpe")
    
    print("Optimal Weights:")
    for sym, w in zip(returns.columns, weights):
        print(f"  {sym}: {w:.2%}")
    
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {metrics['expected_return']:.2%}")
    print(f"  Volatility: {metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
