import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Button, Paper } from '@mui/material';

const UserPredict = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    const previewURL = URL.createObjectURL(file);
    setImagePreview(previewURL);
    // Reset prediction
    setPrediction(null);
  };

  const handlePredict = () => {
    const formData = new FormData();
    formData.append('image', selectedFile);

    axios.post('http://127.0.0.1:8000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
      .then(response => {
        console.log(response);
        setPrediction(response.data.score);
      })
      .catch(error => {
        console.error('Error predicting:', error);
      });
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <Paper elevation={3} style={{ padding: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center', backgroundColor: '#D9D9D9' }}>
        <h2>Make A Prediction</h2>
        <label htmlFor="file-upload" style={{ marginBottom: '20px' }}>
          <Button variant="contained" style={{ backgroundColor: '#1A353E', color: 'white' }} onClick={triggerFileInput}>
            Choose file
          </Button>
          <input type="file" id="file-upload" onChange={handleFileChange} style={{ display: 'none' }} ref={fileInputRef} />
        </label>
        {imagePreview && <img src={imagePreview} alt="Selected File" style={{ maxWidth: '100%', marginTop: '10px' }} />}
        <Button variant="contained" onClick={handlePredict} style={{ backgroundColor: '#1A353E', color: 'white', marginTop: '20px' }}>
          Predict
        </Button>
        {prediction && <div><p>Prediction: {prediction}</p></div>}
      </Paper>
    </div>
  );
};

export default UserPredict;
