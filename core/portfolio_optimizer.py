import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from sklearn.covariance import EmpiricalCovariance, MinCovDet

class PortfolioOptimizer:
    """Advanced portfolio optimization with risk management and constraints."""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 max_position_size: float = 0.3,
                 min_position_size: float = 0.0,
                 target_volatility: float = 0.15,
                 rebalance_threshold: float = 0.1):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            max_position_size: Maximum allocation for any single asset
            min_position_size: Minimum allocation for any single asset
            target_volatility: Target portfolio volatility
            rebalance_threshold: Threshold for triggering rebalance
        """
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.target_volatility = target_volatility
        self.rebalance_threshold = rebalance_threshold
        
    def calculate_optimal_weights(self,
                                returns: pd.DataFrame,
                                current_weights: Optional[np.ndarray] = None,
                                constraints: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Calculate optimal portfolio weights using mean-variance optimization.
        
        Args:
            returns: DataFrame of asset returns
            current_weights: Current portfolio weights (if rebalancing)
            constraints: Additional portfolio constraints
            
        Returns:
            optimal_weights: Array of optimal portfolio weights
            metrics: Dictionary of portfolio metrics
        """
        try:
            # Calculate expected returns and covariance
            exp_returns = self._calculate_expected_returns(returns)
            cov_matrix = self._calculate_robust_covariance(returns)
            
            # Initial weights if not provided
            n_assets = len(returns.columns)
            if current_weights is None:
                current_weights = np.array([1/n_assets] * n_assets)
            
            # Define optimization constraints
            constraints_list = self._create_constraints(n_assets, constraints)
            
            # Define bounds for each asset
            bounds = tuple((self.min_position_size, self.max_position_size) for _ in range(n_assets))
            
            # Optimize portfolio
            result = minimize(
                fun=self._portfolio_objective,
                x0=current_weights,
                args=(exp_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            
            if not result.success:
                raise Exception(f"Optimization failed: {result.message}")
            
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(
                optimal_weights, exp_returns, cov_matrix
            )
            
            return optimal_weights, metrics
            
        except Exception as e:
            print(f"Error in portfolio optimization: {e}")
            return current_weights, {}
    
    def _calculate_expected_returns(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate expected returns using exponential weighted average."""
        return returns.ewm(span=252).mean().iloc[-1].values
    
    def _calculate_robust_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate robust covariance matrix using Minimum Covariance Determinant."""
        robust_cov = MinCovDet(random_state=42)
        robust_cov.fit(returns)
        return robust_cov.covariance_
    
    def _portfolio_objective(self,
                           weights: np.ndarray,
                           exp_returns: np.ndarray,
                           cov_matrix: np.ndarray) -> float:
        """
        Portfolio objective function to maximize Sharpe Ratio.
        """
        portfolio_return = np.sum(weights * exp_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Adjust for risk-free rate
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Minimize negative Sharpe Ratio (maximize Sharpe Ratio)
        return -sharpe
    
    def _create_constraints(self,
                          n_assets: int,
                          additional_constraints: Optional[Dict] = None) -> List:
        """Create optimization constraints."""
        constraints = [
            # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            
            # Target volatility constraint
            {
                'type': 'ineq',
                'fun': lambda x, cov: self.target_volatility - np.sqrt(np.dot(x.T, np.dot(cov, x))),
                'args': (self._calculate_robust_covariance,)
            }
        ]
        
        if additional_constraints:
            # Add sector/asset class constraints
            if 'sector_limits' in additional_constraints:
                for sector, (assets, limit) in additional_constraints['sector_limits'].items():
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, assets=assets: limit - np.sum(x[assets])
                    })
            
            # Add turnover constraints
            if 'max_turnover' in additional_constraints:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x: additional_constraints['max_turnover'] - np.sum(np.abs(x - np.array(additional_constraints['current_weights'])))
                })
        
        return constraints
    
    def _calculate_portfolio_metrics(self,
                                   weights: np.ndarray,
                                   exp_returns: np.ndarray,
                                   cov_matrix: np.ndarray) -> Dict:
        """Calculate portfolio performance metrics."""
        portfolio_return = np.sum(weights * exp_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Calculate diversification ratio
        asset_vols = np.sqrt(np.diag(cov_matrix))
        weighted_vols = weights * asset_vols
        div_ratio = np.sum(weighted_vols) / portfolio_vol
        
        # Calculate risk contribution
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contribution = weights * marginal_risk
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': div_ratio,
            'risk_contribution': risk_contribution
        }
    
    def check_rebalance_needed(self,
                              current_weights: np.ndarray,
                              target_weights: np.ndarray) -> bool:
        """Check if portfolio rebalancing is needed."""
        return np.any(np.abs(current_weights - target_weights) > self.rebalance_threshold)
    
    def generate_rebalancing_trades(self,
                                  current_weights: np.ndarray,
                                  target_weights: np.ndarray,
                                  portfolio_value: float) -> pd.DataFrame:
        """Generate trades needed to rebalance portfolio."""
        weight_diff = target_weights - current_weights
        trade_values = weight_diff * portfolio_value
        
        trades = pd.DataFrame({
            'current_weight': current_weights,
            'target_weight': target_weights,
            'weight_diff': weight_diff,
            'trade_value': trade_values
        })
        
        return trades[np.abs(trades['weight_diff']) > 0.001]  # Filter out tiny trades
