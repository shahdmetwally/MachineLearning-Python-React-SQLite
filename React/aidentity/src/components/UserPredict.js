import React, { useState } from 'react';
import axios from 'axios';

const UserPredict = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handlePredict = () => {
      const formData = new FormData();
      formData.append('image', selectedFile);

      axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: {
            "Content-Type": "multipart/form-data",
          }
      })
      .then(response => {
        console.log(response);
        setPrediction(response.data.score);
      })
      .catch(error => {
        console.error('Error predicting:', error);
      });
    };
    console.log('Prediction:', prediction);
  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handlePredict}>Predict</button>
      {prediction && <p>Prediction Score: {prediction}</p>}
    </div>
  );
};

export default UserPredict;