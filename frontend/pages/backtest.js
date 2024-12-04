import { useEffect, useState } from 'react';
import BacktestVisualization from '../components/BacktestVisualization';

export default function BacktestPage() {
  const [backtestData, setBacktestData] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [timeframe, setTimeframe] = useState('1h');

  useEffect(() => {
    fetch('/api/strategies')
      .then(res => res.json())
      .then(data => {
        setStrategies(data);
        if (data.length > 0) setSelectedStrategy(data[0]);
      });
  }, []);

  useEffect(() => {
    if (selectedStrategy) {
      fetch(`/api/backtest-results?strategy=${selectedStrategy}&timeframe=${timeframe}`)
        .then(res => res.json())
        .then(data => setBacktestData(data));
    }
  }, [selectedStrategy, timeframe]);

  return (
    <div className="container mx-auto p-4">
      <div className="mb-4 flex space-x-4">
        <select
          value={selectedStrategy}
          onChange={(e) => setSelectedStrategy(e.target.value)}
          className="border p-2 rounded"
        >
          {strategies.map(strategy => (
            <option key={strategy} value={strategy}>{strategy}</option>
          ))}
        </select>
        
        <select
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="1m">1 minute</option>
          <option value="5m">5 minutes</option>
          <option value="15m">15 minutes</option>
          <option value="1h">1 hour</option>
          <option value="4h">4 hours</option>
          <option value="1d">1 day</option>
        </select>
      </div>
      
      {backtestData && <BacktestVisualization data={backtestData} />}
    </div>
  );
}