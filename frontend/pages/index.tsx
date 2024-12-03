import Head from 'next/head';
import { useEffect, useState } from 'react';
import { Container, Grid, Card, CardContent, Typography, Button } from '@mui/material';
import { useRouter } from 'next/router';

interface Strategy {
  id: string;
  name: string;
  description: string;
  status: string;
}

export default function Home() {
  const router = useRouter();
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('http://localhost:5001/api/strategies')
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => setStrategies(data))
      .catch(error => {
        console.error('Error fetching strategies:', error);
        setError(error.message);
      });
  }, []);

  const handleViewDetails = (strategyId: string) => {
    router.push(`/strategy/${strategyId}`);
  };

  return (
    <div>
      <Head>
        <title>Crypto Trading Bot</title>
        <meta name="description" content="Monitor your crypto trading strategies" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Container sx={{ py: 4 }}>
        <Typography variant="h3" align="center" gutterBottom>
          Crypto Trading Strategies
        </Typography>

        {error && (
          <Typography color="error" align="center" gutterBottom>
            Error: {error}
          </Typography>
        )}

        <Grid container spacing={4}>
          {strategies.map(strategy => (
            <Grid item xs={12} sm={6} md={4} key={strategy.id}>
              <Card>
                <CardContent>
                  <Typography variant="h5" component="div" gutterBottom>
                    {strategy.name}
                  </Typography>
                  <Typography color="text.secondary" paragraph>
                    {strategy.description}
                  </Typography>
                  <Typography color="text.secondary" gutterBottom>
                    Status: <span style={{ color: strategy.status === 'active' ? 'green' : 'red' }}>{strategy.status}</span>
                  </Typography>
                  <Button 
                    variant="contained" 
                    color="primary"
                    onClick={() => handleViewDetails(strategy.id)}
                  >
                    View Details
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
    </div>
  );
}
