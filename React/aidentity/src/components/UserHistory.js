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
    <li>
      <p>Score: {prediction.score}</p>
      {imageLoaded && <img src={`http://127.0.0.1:8000/get_image/${prediction.image}`} alt="Prediction" style={{ maxWidth: '100%' }} />}
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
    <Paper elevation={0} style={{ backgroundColor: '#1A353E', marginRight: '20px', padding: '20px' }}>
      <h2 style={{ color: '#D9D9D9', marginBottom: '10px', textAlign: 'center' }}>History</h2>
      <Paper elevation={3} style={{ padding: '20px', backgroundColor: '#D9D9D9' }}>
        <ul>
          {predictions.map((prediction, index) => (
            <PredictionItem key={index} prediction={prediction} />
          ))}
        </ul>
      </Paper>
    </Paper>
  );
};

export default UserHistory;
