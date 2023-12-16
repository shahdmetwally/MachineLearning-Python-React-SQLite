import React from "react";
import UserHistory from "../components/UserHistory";
import UserPredict from "../components/UserPredict";
import { Paper } from '@mui/material';

function Home() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', maxWidth: '1000px', margin: '0 auto', marginLeft: '10px'  }}>
      <Paper style={{ padding: '20px', backgroundColor: '#6E99A7', marginRight: '20px', outline: 'none', marginTop: '20px', boxShadow: 'none' }}>
        <UserHistory />
      </Paper>
      <Paper style={{ padding: '20px', backgroundColor: '#6E99A7', marginRight: '20px', outline: 'none', marginTop: '50px', boxShadow: 'none' }}>
        <UserPredict />
      </Paper>
    </div>
  );
}

export default Home;
