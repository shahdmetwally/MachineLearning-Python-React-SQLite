import React, { useState } from 'react';
import axios from 'axios';

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
      <button onClick={fetchPredictions}>Show History</button>
      <ul>
        {predictions.map((prediction, index) => (
          <li key={index}>{JSON.stringify(prediction)}</li>
          // Render each prediction item as needed
        ))}
      </ul>
    </div>
  );
};

export default UserHistory;
