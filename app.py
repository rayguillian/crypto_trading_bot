from flask import Flask, jsonify, request
from core.backtest import load_backtest_results

app = Flask(__name__)

@app.route('/api/backtest-results', methods=['GET'])
def get_backtest_results():
    strategy_name = request.args.get('strategy')
    timeframe = request.args.get('timeframe')
    results = load_backtest_results(strategy_name, timeframe)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
