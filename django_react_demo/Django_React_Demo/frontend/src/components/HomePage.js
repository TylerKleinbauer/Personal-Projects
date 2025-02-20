import React, {useEffect} from 'react';
import { Button, Grid, Typography, ButtonGroup } from '@mui/material';
import { useNavigate } from 'react-router-dom';

function HomePage() {
  const navigate = useNavigate();

  useEffect(() => {
    fetch('/api/user-in-room')
    .then((response) => response.json())
    .then((data) => {
      if (data.code) {
        navigate('/room/' + data.code);
      }
    });
  });
  
  return (
    <Grid container spacing={3}>
      <Grid item xs={12} align="center">
        <Typography variant="h3" compact="h3">
          House Party
        </Typography>
      </Grid>
      <Grid item xs={12} align="center">
        <ButtonGroup disableElevation variant="contained" color="primary">
          <Button color="primary" onClick={() => navigate("/join")}>
            Join a Room
          </Button>
          <Button color="secondary" onClick={() => navigate("/create")}>
            Create a Room
          </Button>
        </ButtonGroup>
      </Grid>
    </Grid>
  );
}

export default HomePage;
