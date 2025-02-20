import React, {component} from 'react';
import {Grid, Typography, Button, IconButton, LinearProgress, Card} from '@mui/material';
import {Pause, PlayArrow, SkipNext, SkipPrevious} from '@mui/icons-material';

export default function MusicPlayer(props) {
  const songProgress = (props.time / props.duration) * 100;
  
  const pauseSong = () => {
    const requestOptions = {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
    };
    fetch("/spotify/pause", requestOptions);
  };

  const playSong = () => {
    const requestOptions = {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
    };
    fetch("/spotify/play", requestOptions);
  };

  const skipSong = (forward = true) => {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    };
    if (forward) {
        fetch("/spotify/skip", requestOptions);
    } else {
        fetch("/spotify/previous", requestOptions);
    }
  };

  return (
    <Card>
      <Grid container alignItems="center">
        <Grid item align="center" xs={4}>
          <img src={props.image_url} height="100%" width="100%" />
        </Grid>
        <Grid item align="center" xs={8}>
          <Typography component="h5" variant="h5">
            {props.title}
          </Typography>
          <Typography color="textSecondary" variant="subtitle1">
            {props.artist}
          </Typography>
          <div>
            <IconButton onClick={() => skipSong(false)}>
              <SkipPrevious />
            </IconButton>
            <IconButton onClick={props.is_playing ? pauseSong : playSong}>
              {props.is_playing ? <Pause /> : <PlayArrow />}
            </IconButton>
            <IconButton onClick={() => skipSong(true)}>
              <SkipNext /> {" "}{props.votes} / {" "}{props.votes_required}
            </IconButton>
          </div>
        </Grid>
      </Grid>
      <LinearProgress variant="determinate" value={songProgress} />
    </Card>
  );
}