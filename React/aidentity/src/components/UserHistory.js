import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Paper } from '@mui/material';

const PredictionItem = ({ prediction }) => {
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    const img = new Image();
    img.src = `http://127.0.0.1:8000/get_image/${prediction.image}`;
    img.onload = () => setImageLoaded(true);
    img.onerror = (error) => console.error('Error loading image:', error);
  }, [prediction.image]);

  return (
    <li style={{ marginBottom: '15px' }}>
      <p>Name: {prediction.score}</p>
      {imageLoaded && (
        <img
          src={`http://127.0.0.1:8000/get_image/${prediction.image}`}
          alt="Prediction"
          style={{ maxWidth: '100%', maxHeight: '100px', objectFit: 'contain' }}
        />
      )}
      <p>Created At: {prediction.created_at}</p>
    </li>
  );
};

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
            {predictions.map((prediction, index) => (
              <PredictionItem key={index} prediction={prediction} />
            ))}
          </ul>
        </Paper>
      </Paper>
    </div>
  );
};

export default UserHistory;
