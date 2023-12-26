import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Paper } from '@mui/material';


const UserHistory = () => {
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/predictions');
        setPredictions(response.data);
      } catch (error) {
        console.error('Error while fetching predictions:', error);
      }
    };
    
    fetchPredictions();
  }, []); // Empty array ensures useEffect runs only once (on mount)

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', maxWidth: '1000px', margin: '0 auto', marginLeft: '10px' }}>
      <Paper elevation={3} style={{ padding: '20px', backgroundColor: '#1A353E', width: '100%' }}>
        <h2 style={{ color: '#D9D9D9', marginBottom: '10px', textAlign: 'center' }}>History</h2>
        <Paper elevation={3} style={{ maxHeight: '400px', overflowY: 'auto', padding: '20px', backgroundColor: '#D9D9D9' }}>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {predictions.map(prediction => (
              <li key={prediction.id} style={{ marginBottom: '20px' }}>
                <p>Score: {prediction.score}</p>
                <p>Created At: {prediction.created_at}</p>
                <img src={`data:image/jpeg;base64,${prediction.image}`} alt="Prediction" style={{ width: '100%', height: '200px', objectFit: 'cover' }} />
              </li>
            ))}
          </ul>
        </Paper>
      </Paper>
    </div>
  );
};

export default UserHistory;
