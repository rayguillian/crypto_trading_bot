from flask import Flask, jsonify, request, send_from_directory
from core.backtest import load_backtest_results
import os

app = Flask(__name__, static_folder='frontend/out', static_url_path='')

@app.route('/api/backtest-results', methods=['GET'])
def get_backtest_results():
    strategy_name = request.args.get('strategy')
    timeframe = request.args.get('timeframe')
    results = load_backtest_results(strategy_name, timeframe)
    return jsonify(results)

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    strategy_dir = 'strategy_results'
    strategies = [d for d in os.listdir(strategy_dir) 
                 if os.path.isdir(os.path.join(strategy_dir, d))]
    return jsonify(strategies)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)