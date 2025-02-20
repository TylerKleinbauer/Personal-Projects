import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Button, Grid, Typography } from '@mui/material';
import CreateRoomPage from "./CreateRoomPage";
import MusicPlayer from "./MusicPlayer";

export default function Room() {
  const navigate = useNavigate();
  const { roomCode } = useParams(); // Accessing the room code from the URL
  const [state, setState] = useState({
    votesToSkip: 2,
    guestCanPause: false,
    isHost: false,
    spotifyAuthenticated: false,
    song: {},
  });
  const [roomExists, setRoomExists] = useState(true); // This state will be used to check if the room exists
  const [loading, setLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false); // State to manage loading state

  const getRoomDetails = async () => {
    try {
      const response = await fetch("/api/get-room?code=" + roomCode);
      if (!response.ok) {
        setRoomExists(false);
        setLoading(false);
        return;
      }
      const data = await response.json();
      setState((prevState) => ({
        ...prevState,
        votesToSkip: data.votes_to_skip,
        guestCanPause: data.guest_can_pause,
        isHost: data.is_host,
      }));
      setLoading(false);
    } catch (error) {
      console.error("Error fetching room details:", error);
      setRoomExists(false);
      setLoading(false);
    }
  };

  useEffect(() => {  // Use the useEffect hook to fetch the room details when the component mounts
    getRoomDetails();
  },[roomCode]);   // This will run the effect whenever the roomCode changes

  useEffect(() => {
    if (state.isHost && !state.spotifyAuthenticated) {
      authenticate_spotify();
    }
  }, [state.isHost, state.spotifyAuthenticated]);
  
  useEffect(() => {
    const interval = setInterval(() => {
      getCurrentSong();
    }, 1000);
    return () => clearInterval(interval); // This will clear the interval when the component unmounts
  }, [state.spotifyAuthenticated]);       // This will run the effect whenever spotifyAuthenticated changes
  
  const handleBackClick = () => {
    navigate("/");
  };

  const authenticate_spotify = () => {
    fetch("/spotify/is-authenticated")
      .then((response) => response.json())
      .then((data) => {
        setState((prevState) => ({
          ...prevState,
          spotifyAuthenticated: data.status,
        }));
        if (!data.status) {
          fetch("/spotify/get-auth-url")
            .then((response) => response.json())
            .then((data) => {
              window.location.replace(data.url);
            });
        }
      });
  };

  const getCurrentSong = () => {
    fetch("/spotify/current-song")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to fetch current song.");
        } else if (response.status === 204) {
          return null;
        } else {
          return response.json();
        }
      })
      .then((data) => {
        if (data) {
          setState((prevState) => ({
            ...prevState,
            song: data,
          }));
          if (data) {
            console.log(data);
          } else {
            console.log("No song currently playing");
          }
        }
      })
      .catch((error) => {
        console.error("Error fetching current song:", error);
      });
  };

  const leaveButtonPressed = () => {
    fetch("/api/leave-room", {
      method: "POST",
    }).then((_response) => {
      navigate("/");
    });
  };

  const renderSettings = () => {
    return (
      <Grid container spacing={1}>
        <Grid item xs={12} align="center">
          <CreateRoomPage
            update={true}
            votesToSkip={state.votesToSkip}
            guestCanPause={state.guestCanPause}
            roomCode={roomCode}
            updateCallback={getRoomDetails}
          />
        </Grid>
        <Grid item xs={12} align="center">
          <Button
            variant="contained"
            color="secondary"
            onClick={() => setShowSettings(false)}
          >
            Close
          </Button>
        </Grid>
      </Grid>
    );
  };

  const renderSettingsButton = () => {
    return (
      <Grid item xs={12} align="center">
        <Button
          variant="contained"
          color="primary"
          onClick={() => setShowSettings(true)}
        >
          Settings
        </Button>
      </Grid>
    );
  };

  if (loading) {
    return (
      <Grid container spacing={1}>
        <Grid item xs={12} align="center">
          <Typography variant="h5" component="h5">
            Loading...
          </Typography>
        </Grid>
      </Grid>
    );
  }

  if (!roomExists) {
    return (
      <Grid container spacing={1}>
        <Grid item xs={12} align="center">
          <Typography variant="h5" component="h5">
            Room not found.
          </Typography>
        </Grid>
        <Grid item xs={12} align="center">
          <Button
            color="secondary"
            variant="contained"
            onClick={handleBackClick}
          >
            Back to Home
          </Button>
        </Grid>
      </Grid>
    );
  }

  if (showSettings) {
    return renderSettings();
  }

  return (
    <Grid container spacing={1}>
      <Grid item xs={12} align="center">
        <Typography variant="h4" component="h4">
          Room: {roomCode}
        </Typography>
      </Grid>
      <MusicPlayer{...state.song} />
      {state.isHost ? renderSettingsButton() : null}
      <Grid item xs={12} align="center">
        <Button
          color="secondary"
          variant="contained"
          onClick={leaveButtonPressed}
        >
          Leave Room
        </Button>
      </Grid>
    </Grid>
  );
}
