// Strategy Development Workflow
let currentStep = 1;
let totalSteps = 4;
let strategyData = {};

// Initialize strategy development
function initStrategyDevelopment() {
    currentStep = 1;
    strategyData = {};
    updateProgressBar();
    showStep(1);
}

// Update progress bar
function updateProgressBar() {
    const progress = ((currentStep - 1) / (totalSteps - 1)) * 100;
    $('.progress-bar').css('width', `${progress}%`);
}

// Show specific step
function showStep(step) {
    $('.step-content').addClass('d-none');
    $(`#step${step}`).removeClass('d-none');
    
    // Update buttons
    if (step === 1) {
        $('button:contains("Previous")').addClass('d-none');
    } else {
        $('button:contains("Previous")').removeClass('d-none');
    }
    
    if (step === totalSteps) {
        $('button:contains("Next")').addClass('d-none');
        $('#finishButton').removeClass('d-none');
    } else {
        $('button:contains("Next")').removeClass('d-none');
        $('#finishButton').addClass('d-none');
    }
}

// Navigate to next step
function nextStep() {
    if (currentStep < totalSteps) {
        if (validateStep(currentStep)) {
            currentStep++;
            updateProgressBar();
            showStep(currentStep);
            
            // Load step-specific content
            if (currentStep === 2) {
                loadBacktestingParameters();
            }
        }
    }
}

// Navigate to previous step
function previousStep() {
    if (currentStep > 1) {
        currentStep--;
        updateProgressBar();
        showStep(currentStep);
    }
}

// Validate current step
function validateStep(step) {
    if (step === 1) {
        const name = $('#strategyName').val();
        const type = $('#strategyType').val();
        const symbol = $('#tradingPair').val();
        
        if (!name || !type || !symbol) {
            alert('Please fill in all required fields');
            return false;
        }
        
        strategyData = {
            name: name,
            type: type,
            symbol: symbol,
            interval: $('#timeframe').val()
        };
    }
    return true;
}

// Load strategy parameters based on type
function loadBacktestingParameters() {
    const type = $('#strategyType').val();
    let params = '';
    
    if (type === 'Momentum') {
        params = `
            <div class="mb-3">
                <label class="form-label">RSI Period</label>
                <input type="number" class="form-control" id="rsiPeriod" value="14" min="2" max="50">
            </div>
            <div class="mb-3">
                <label class="form-label">RSI Overbought</label>
                <input type="number" class="form-control" id="rsiOverbought" value="70" min="50" max="90">
            </div>
            <div class="mb-3">
                <label class="form-label">RSI Oversold</label>
                <input type="number" class="form-control" id="rsiOversold" value="30" min="10" max="50">
            </div>
        `;
    } else if (type === 'Mean Reversion') {
        params = `
            <div class="mb-3">
                <label class="form-label">Bollinger Band Period</label>
                <input type="number" class="form-control" id="bbPeriod" value="20" min="5" max="50">
            </div>
            <div class="mb-3">
                <label class="form-label">Standard Deviation</label>
                <input type="number" class="form-control" id="bbStd" value="2" min="1" max="3" step="0.1">
            </div>
        `;
    }
    
    $('#strategyParams').html(params);
}

// Run backtest
async function runBacktest() {
    const params = getStrategyParameters();
    const lookbackDays = $('#lookbackDays').val();
    
    try {
        const response = await fetch('/api/strategy/develop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ...strategyData,
                lookback_days: parseInt(lookbackDays),
                parameters: params
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBacktestResults(data.strategy.backtesting_results);
            strategyData.id = data.strategy.id;
        } else {
            alert('Backtest failed: ' + data.error);
        }
    } catch (error) {
        console.error('Backtest error:', error);
        alert('Failed to run backtest');
    }
}

// Get strategy parameters based on type
function getStrategyParameters() {
    const type = $('#strategyType').val();
    let params = {};
    
    if (type === 'Momentum') {
        params = {
            rsi_period: parseInt($('#rsiPeriod').val()),
            rsi_overbought: parseInt($('#rsiOverbought').val()),
            rsi_oversold: parseInt($('#rsiOversold').val())
        };
    } else if (type === 'Mean Reversion') {
        params = {
            bb_period: parseInt($('#bbPeriod').val()),
            bb_std: parseFloat($('#bbStd').val())
        };
    }
    
    return params;
}

// Display backtest results
function displayBacktestResults(results) {
    $('#backtestResults').removeClass('d-none');
    
    // Performance metrics
    const metrics = `
        <p>Total Return: ${(results.total_returns * 100).toFixed(2)}%</p>
        <p>Sharpe Ratio: ${results.sharpe_ratio.toFixed(2)}</p>
        <p>Max Drawdown: ${(results.max_drawdown * 100).toFixed(2)}%</p>
    `;
    $('#performanceMetrics').html(metrics);
    
    // Trade statistics
    const stats = `
        <p>Win Rate: ${(results.win_rate * 100).toFixed(2)}%</p>
        <p>Total Trades: ${results.trades}</p>
    `;
    $('#tradeStats').html(stats);
}

// Optimize strategy
async function optimizeStrategy() {
    try {
        const response = await fetch(`/api/strategy/${strategyData.id}/optimize`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayOptimizationResults(data.optimization_results);
        } else {
            alert('Optimization failed: ' + data.error);
        }
    } catch (error) {
        console.error('Optimization error:', error);
        alert('Failed to optimize strategy');
    }
}

// Display optimization results
function displayOptimizationResults(results) {
    $('#optimizationResults').removeClass('d-none');
    
    let paramsHtml = '<ul class="list-unstyled">';
    for (const [key, value] of Object.entries(results.params)) {
        paramsHtml += `<li>${key}: ${value}</li>`;
    }
    paramsHtml += '</ul>';
    
    $('#optimizedParams').html(paramsHtml);
}

// Finish strategy development
async function finishStrategyDevelopment() {
    const initialCapital = $('#initialCapital').val();
    const riskPerTrade = $('#riskPerTrade').val();
    const stopLossType = $('#stopLossType').val();
    
    try {
        const response = await fetch(`/api/strategy/${strategyData.id}/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                initial_capital: parseFloat(initialCapital),
                risk_per_trade: parseFloat(riskPerTrade),
                stop_loss_type: stopLossType
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            $('#strategyModal').modal('hide');
            location.reload(); // Refresh to show new strategy
        } else {
            alert('Failed to start strategy: ' + data.error);
        }
    } catch (error) {
        console.error('Start strategy error:', error);
        alert('Failed to start strategy');
    }
}

// Update real-time market data
async function updateMarketData() {
    try {
        const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT'];
        const prices = await Promise.all(symbols.map(async (symbol) => {
            const response = await fetch(`/api/market/price/${symbol}`);
            return response.json();
        }));
        
        // Update price displays
        prices.forEach((data, index) => {
            const symbol = symbols[index].replace('USDT', '').toLowerCase();
            if (data.success) {
                $(`#${symbol}Price`).text(`$${parseFloat(data.price).toLocaleString()}`);
            }
        });
    } catch (error) {
        console.error('Failed to update market data:', error);
    }
}

// Update AutoTrader status
async function updateAutoTraderStatus() {
    try {
        const response = await fetch('/api/auto_trader/status');
        const data = await response.json();
        
        if (data.success) {
            $('#autoTraderStatus').text(data.status.toUpperCase());
            $('#activeStrategies').text(data.active_strategies);
            $('#totalStrategies').text(data.total_strategies);
            $('#recentTrades').text(data.recent_trades);
            
            // Update button states
            if (data.status === 'running') {
                $('#startAutoTrader').prop('disabled', true);
                $('#stopAutoTrader').prop('disabled', false);
            } else {
                $('#startAutoTrader').prop('disabled', false);
                $('#stopAutoTrader').prop('disabled', true);
            }
        } else {
            $('#autoTraderStatus').text('ERROR');
        }
    } catch (error) {
        console.error('Failed to update AutoTrader status:', error);
        $('#autoTraderStatus').text('ERROR');
    }
}

// Start AutoTrader
async function startAutoTrader() {
    try {
        const response = await fetch('/api/auto_trader/start', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            updateAutoTraderStatus();
        } else {
            alert('Failed to start AutoTrader: ' + data.error);
        }
    } catch (error) {
        console.error('Failed to start AutoTrader:', error);
        alert('Failed to start AutoTrader');
    }
}

// Stop AutoTrader
async function stopAutoTrader() {
    try {
        const response = await fetch('/api/auto_trader/stop', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            updateAutoTraderStatus();
        } else {
            alert('Failed to stop AutoTrader: ' + data.error);
        }
    } catch (error) {
        console.error('Failed to stop AutoTrader:', error);
        alert('Failed to stop AutoTrader');
    }
}

// Event listeners
$(document).ready(function() {
    refreshData();
    // Refresh every 30 seconds
    setInterval(refreshData, 30000);
    
    // Start real-time market data updates
    updateMarketData();
    // Update market data every 5 seconds
    setInterval(updateMarketData, 5000);
    
    $('#strategyModal').on('show.bs.modal', function() {
        initStrategyDevelopment();
    });
    
    $('#strategyType').on('change', function() {
        if (currentStep === 2) {
            loadBacktestingParameters();
        }
    });
    
    // AutoTrader controls
    $('#startAutoTrader').click(startAutoTrader);
    $('#stopAutoTrader').click(stopAutoTrader);
    
    // Update AutoTrader status every 10 seconds
    updateAutoTraderStatus();
    setInterval(updateAutoTraderStatus, 10000);
});
