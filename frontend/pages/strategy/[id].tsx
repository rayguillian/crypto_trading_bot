import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import { Container, Typography, Card, CardContent, Grid, Button, CircularProgress, Alert } from '@mui/material';

interface StrategyDetails {
  id: string;
  name: string;
  description: string;
  status: string;
  performance?: {
    total_return: number;
    win_rate: number;
    sharpe_ratio: number;
    max_drawdown: number;
    total_trades: number;
  };
  parameters?: Record<string, any>;
}

export default function StrategyDetails() {
  const router = useRouter();
  const { id } = router.query;
  const [strategy, setStrategy] = useState<StrategyDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    setLoading(true);
    fetch(`http://localhost:5001/api/strategies/${id}`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        setStrategy(data);
        setError(null);
      })
      .catch(error => {
        console.error('Error fetching strategy details:', error);
        setError(error.message);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <Container sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (error) {
    return (
      <Container sx={{ py: 4 }}>
        <Alert severity="error">Error loading strategy details: {error}</Alert>
        <Button sx={{ mt: 2 }} variant="contained" onClick={() => router.push('/')}>
          Back to Strategies
        </Button>
      </Container>
    );
  }

  if (!strategy) {
    return (
      <Container sx={{ py: 4 }}>
        <Alert severity="warning">Strategy not found</Alert>
        <Button sx={{ mt: 2 }} variant="contained" onClick={() => router.push('/')}>
          Back to Strategies
        </Button>
      </Container>
    );
  }

  return (
    <div>
      <Head>
        <title>{strategy.name} - Crypto Trading Bot</title>
        <meta name="description" content={`Details for ${strategy.name}`} />
      </Head>

      <Container sx={{ py: 4 }}>
        <Button variant="outlined" onClick={() => router.push('/')} sx={{ mb: 4 }}>
          Back to Strategies
        </Button>

        <Typography variant="h4" component="h1" gutterBottom>
          {strategy.name}
        </Typography>

        <Grid container spacing={4}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Overview</Typography>
                <Typography paragraph>{strategy.description}</Typography>
                <Typography color="text.secondary">
                  Status: <span style={{ color: strategy.status === 'active' ? 'green' : 'red' }}>{strategy.status}</span>
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {strategy.performance && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6} md={4}>
                      <Typography color="text.secondary">Total Return</Typography>
                      <Typography variant="h6">{(strategy.performance.total_return * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Typography color="text.secondary">Win Rate</Typography>
                      <Typography variant="h6">{(strategy.performance.win_rate * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Typography color="text.secondary">Sharpe Ratio</Typography>
                      <Typography variant="h6">{strategy.performance.sharpe_ratio.toFixed(2)}</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Typography color="text.secondary">Max Drawdown</Typography>
                      <Typography variant="h6">{(strategy.performance.max_drawdown * 100).toFixed(2)}%</Typography>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Typography color="text.secondary">Total Trades</Typography>
                      <Typography variant="h6">{strategy.performance.total_trades}</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          )}

          {strategy.parameters && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Strategy Parameters</Typography>
                  <Grid container spacing={2}>
                    {Object.entries(strategy.parameters).map(([key, value]) => (
                      <Grid item xs={12} sm={6} md={4} key={key}>
                        <Typography color="text.secondary">{key}</Typography>
                        <Typography variant="body1">{value}</Typography>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Container>
    </div>
  );
}
