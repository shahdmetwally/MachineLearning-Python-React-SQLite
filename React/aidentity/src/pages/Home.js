import React from "react";
import UserHistory from "../components/UserHistory";
import UserPredict from "../components/UserPredict";
import { Grid, Paper } from "@mui/material";

function Home() {
  return (
    <Grid
      container
      spacing={3}
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "90vh",
        position: "absolute"
      }}
    >
      <Grid item>
        <Paper
          style={{
            padding: "20px",
            backgroundColor: "#6E99A7",
            outline: "none",
            boxShadow: "none",
          }}
        >
          <UserHistory />
        </Paper>
      </Grid>
      <Grid item>
        <Paper
          style={{
            padding: "20px",
            backgroundColor: "#6E99A7",
            outline: "none",
            boxShadow: "none",
          }}
        >
          <UserPredict />
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Home;
