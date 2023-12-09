import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from '@mui/material';

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

  const fetchPredictions = () => {
    axios.get('http://127.0.0.1:8000/predictions')
      .then(response => {
        // Set predictions
        setPredictions(response.data);
      })
      .catch(error => {
        console.error('Error while fetching predictions:', error);
      });
  };

  return (
    <div>
      <h2>User Prediction History</h2>
      <Button
        variant="contained" onClick={fetchPredictions} style={{ backgroundColor: '#1A353E', color: 'white' }}>Show History</Button>
      <ul>
        {predictions.map((prediction, index) => (
          <PredictionItem key={index} prediction={prediction} />
        ))}
      </ul>
    </div>
  );
};

export default UserHistory;