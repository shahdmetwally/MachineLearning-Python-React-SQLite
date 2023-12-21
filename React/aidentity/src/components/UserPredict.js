import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Button, Paper } from '@mui/material';

const UserPredict = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const fileInputRef = useRef(null);
  const [isPredictionCorrect, setIsPredictionCorrect] = useState(null);
  const [userName, setUserName] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    const previewURL = URL.createObjectURL(file);
    setImagePreview(previewURL);
    setPrediction(null);
    setIsPredictionCorrect(null);
    setUserName('');
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

  const handleFeedback = () => {
    if (isPredictionCorrect !== null) {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('is_correct', isPredictionCorrect);

      if (isPredictionCorrect === 'false' && userName.trim() !== '') {
        formData.append('user_name', userName.trim());
      }

      axios.post('http://127.0.0.1:8000/feedback', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
        .then((response) => {
          console.log(response);
          // Handle the response as needed
        })
        .catch((error) => {
          console.error('Error submitting feedback:', error);
        });
    }
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
        {prediction && (
        <div>
          <p>Prediction Score: {prediction}</p>
          <label>
            Is the prediction correct?
            <select onChange={(e) => setIsPredictionCorrect(e.target.value)}>
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </label>
          {isPredictionCorrect === 'false' && (
            <div>
              <label>
                Enter your name:
                <input type="text" value={userName} onChange={(e) => setUserName(e.target.value)} />
              </label>
              <button onClick={handleFeedback}>Submit Feedback</button>
            </div>
          )}
        </div>
      )}
      </Paper>
</div>
  );
};

export default UserPredict;

