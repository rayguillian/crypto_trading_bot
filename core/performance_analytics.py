import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PerformanceAnalytics:
    """Advanced performance analytics and visualization."""
    
    def __init__(self):
        """Initialize performance analytics."""
        self.metrics = {}
        self.trades = []
        self.equity_curve = None
        
    def analyze_performance(self, 
                          trades: List[Dict],
                          equity_curve: pd.Series,
                          benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Analyze trading performance and calculate metrics.
        
        Args:
            trades: List of trade dictionaries
            equity_curve: Series of portfolio values
            benchmark_returns: Optional benchmark returns series
            
        Returns:
            Dictionary of performance metrics
        """
        self.trades = trades
        self.equity_curve = equity_curve
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        cagr = self._calculate_cagr(equity_curve)
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Trade metrics
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if trades else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if trades else 0
        profit_factor = (
            sum(t['pnl'] for t in trades if t['pnl'] > 0) /
            abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        ) if trades and any(t['pnl'] < 0 for t in trades) else float('inf')
        
        # Advanced metrics
        calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else float('inf')
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else float('inf')
        
        # Risk-adjusted metrics
        if benchmark_returns is not None:
            alpha, beta = self._calculate_alpha_beta(returns, benchmark_returns)
            information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
        else:
            alpha, beta = 0, 1
            information_ratio = 0
            
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'total_trades': len(trades),
            'avg_trade_duration': self._calculate_avg_trade_duration()
        }
        
        return self.metrics
        
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate detailed performance report."""
        report = []
        
        # Overview section
        report.append("# Trading Performance Report")
        report.append(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Returns and risk metrics
        report.append("\n## Returns & Risk Metrics")
        report.append(f"Total Return: {self.metrics['total_return']:.2f}%")
        report.append(f"CAGR: {self.metrics['cagr']:.2f}%")
        report.append(f"Volatility: {self.metrics['volatility']:.2f}%")
        report.append(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        report.append(f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        
        # Trade statistics
        report.append("\n## Trade Statistics")
        report.append(f"Total Trades: {self.metrics['total_trades']}")
        report.append(f"Win Rate: {self.metrics['win_rate']:.2f}%")
        report.append(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        report.append(f"Average Win: ${self.metrics['avg_win']:.2f}")
        report.append(f"Average Loss: ${self.metrics['avg_loss']:.2f}")
        report.append(f"Average Trade Duration: {self.metrics['avg_trade_duration']}")
        
        # Risk-adjusted metrics
        report.append("\n## Risk-Adjusted Metrics")
        report.append(f"Calmar Ratio: {self.metrics['calmar_ratio']:.2f}")
        report.append(f"Recovery Factor: {self.metrics['recovery_factor']:.2f}")
        report.append(f"Alpha: {self.metrics['alpha']:.4f}")
        report.append(f"Beta: {self.metrics['beta']:.2f}")
        report.append(f"Information Ratio: {self.metrics['information_ratio']:.2f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text
        
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot interactive equity curve with drawdown."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=self.equity_curve.index, y=self.equity_curve,
                      name='Portfolio Value',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Drawdown
        drawdown = self._calculate_drawdown_series(self.equity_curve)
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown * 100,
                      name='Drawdown %',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            yaxis2_title='Drawdown (%)',
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def plot_monthly_returns(self, save_path: Optional[str] = None):
        """Plot monthly returns heatmap."""
        # Calculate monthly returns
        monthly_returns = self.equity_curve.resample('M').last().pct_change() * 100
        monthly_returns_table = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).first().unstack()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns_table,
                   annot=True,
                   fmt='.1f',
                   center=0,
                   cmap='RdYlGn',
                   cbar_kws={'label': 'Returns %'})
        
        plt.title('Monthly Returns (%)')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            
    def plot_trade_analysis(self, save_path: Optional[str] = None):
        """Plot trade analysis charts."""
        if not self.trades:
            return
            
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # Create subplots
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Trade PnL Distribution',
                                         'Cumulative PnL',
                                         'Win Rate by Month',
                                         'Average Trade Duration'))
        
        # Trade PnL distribution
        fig.add_trace(
            go.Histogram(x=trades_df['pnl'],
                        name='PnL Distribution',
                        nbinsx=50),
            row=1, col=1
        )
        
        # Cumulative PnL
        fig.add_trace(
            go.Scatter(x=trades_df.index,
                      y=trades_df['pnl'].cumsum(),
                      name='Cumulative PnL'),
            row=1, col=2
        )
        
        # Win rate by month
        monthly_win_rate = (
            trades_df[trades_df['pnl'] > 0]
            .groupby(pd.Grouper(key='exit_time', freq='M'))
            .size()
            / trades_df.groupby(pd.Grouper(key='exit_time', freq='M')).size()
            * 100
        )
        
        fig.add_trace(
            go.Bar(x=monthly_win_rate.index,
                  y=monthly_win_rate.values,
                  name='Monthly Win Rate'),
            row=2, col=1
        )
        
        # Average trade duration
        trades_df['duration'] = (
            pd.to_datetime(trades_df['exit_time']) -
            pd.to_datetime(trades_df['entry_time'])
        ).dt.total_seconds() / 3600  # Convert to hours
        
        fig.add_trace(
            go.Box(y=trades_df['duration'],
                  name='Trade Duration (hours)'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=800, showlegend=False)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def _calculate_cagr(self, equity_curve: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        return (pow(equity_curve[-1] / equity_curve[0], 1/years) - 1) * 100
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino Ratio."""
        negative_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(negative_returns**2))
        return np.sqrt(252) * returns.mean() / downside_std if downside_std > 0 else 0
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min()) * 100
        
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        peak = equity_curve.expanding(min_periods=1).max()
        return (equity_curve - peak) / peak
        
    def _calculate_alpha_beta(self, returns: pd.Series,
                            benchmark_returns: pd.Series) -> tuple:
        """Calculate alpha and beta."""
        # Align series
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        # Calculate beta
        covariance = np.cov(returns, benchmark_returns)[0][1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance if variance > 0 else 1
        
        # Calculate alpha
        alpha = (returns.mean() - beta * benchmark_returns.mean()) * 252
        
        return alpha, beta
        
    def _calculate_information_ratio(self, returns: pd.Series,
                                   benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio."""
        # Align series
        returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
        
        # Calculate tracking error
        tracking_error = (returns - benchmark_returns).std()
        
        # Calculate information ratio
        if tracking_error > 0:
            return (returns.mean() - benchmark_returns.mean()) / tracking_error * np.sqrt(252)
        return 0
        
    def _calculate_avg_trade_duration(self) -> str:
        """Calculate average trade duration."""
        if not self.trades:
            return "N/A"
            
        durations = []
        for trade in self.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = pd.to_datetime(trade['exit_time']) - pd.to_datetime(trade['entry_time'])
                durations.append(duration)
                
        if durations:
            avg_duration = sum(durations, pd.Timedelta(0)) / len(durations)
            hours = avg_duration.total_seconds() / 3600
            return f"{hours:.1f} hours"
        return "N/A"
