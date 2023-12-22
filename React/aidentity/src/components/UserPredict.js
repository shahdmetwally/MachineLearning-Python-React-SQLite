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
  const [videoStream, setVideoStream] = useState(null);
  const videoRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    const previewURL = URL.createObjectURL(file);
    setImagePreview(previewURL);
    setPrediction(null);
    setIsPredictionCorrect(null);
    setUserName('');
  };

  const handlePredictFile = () => {
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

  const handlePredictCamera = async () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
      const timestamp = new Date().getTime(); // Get a unique timestamp
      const fileName = `snapshot_${timestamp}.jpg`; // Use the timestamp in the filename
      const formData = new FormData();
      formData.append('image', blob, fileName);

      try {
        const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        console.log(response);
        setPrediction(response.data.score);
      } catch (error) {
        console.error('Error predicting:', error);
      }
    }, 'image/jpg');
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
        })
        .catch((error) => {
          console.error('Error submitting feedback:', error);
        });
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setVideoStream(stream);
      videoRef.current.srcObject = stream;
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopCamera = () => {
    if (videoStream) {
      const tracks = videoStream.getTracks();
      tracks.forEach((track) => track.stop());
      setVideoStream(null);
    }
  };

  const toggleCamera = () => {
    if (videoStream) {
      stopCamera();
    } else {
      startCamera();
    }
  };
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <Paper elevation={3} style={{ padding: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center', backgroundColor: '#D9D9D9' }}>
        <h2>Make A Prediction</h2>

        {/* Option 1: Upload File */}
        {(!videoStream || selectedFile) && (  // Only show if camera is not started or a file is not selected
          <label htmlFor="file-upload" style={{ marginBottom: '20px' }}>
            <Button variant="contained" style={{ backgroundColor: '#1A353E', color: 'white' }} onClick={triggerFileInput}>
              Choose file
            </Button>
            <input type="file" id="file-upload" onChange={handleFileChange} style={{ display: 'none' }} ref={fileInputRef} />
          </label>
        )}
        {imagePreview && <img src={imagePreview} alt="Selected File" style={{ maxWidth: '100%', marginTop: '10px' }} />}
        {selectedFile && (
          <Button
            variant="contained"
            onClick={handlePredictFile}
            style={{ backgroundColor: '#1A353E', color: 'white', marginTop: '20px' }}
          >
            Predict from File
          </Button>
        )}

        {/* Option 2: Use Camera */}
        {selectedFile ? null : (
          <>
            <video
              ref={videoRef}
              style={{ maxWidth: '100%', marginTop: '10px', display: videoStream ? 'block' : 'none' }}
              autoPlay
            ></video>
            <Button
              variant="contained"
              onClick={toggleCamera}
              style={{ backgroundColor: '#1A353E', color: 'white', marginTop: '20px' }}
            >
              {videoStream ? 'Stop Camera' : 'Start Camera'}
            </Button>
            {videoStream && (
              <Button
                variant="contained"
                onClick={handlePredictCamera}
                style={{ backgroundColor: '#1A353E', color: 'white', marginTop: '20px' }}
              >
                Predict from Camera
              </Button>
            )}
          </>
        )}
        {prediction && (
          <div>
            <p>Prediction Score: {prediction}</p>
            
            {/* Feedback Options */}
            <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <label>
                Is the prediction correct?
              </label>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px' }}>
                <label>
                  <input type="radio" value="true" checked={isPredictionCorrect === 'true'} onChange={() => setIsPredictionCorrect('true')} />
                  Yes
                </label>
                <label>
                  <input type="radio" value="false" checked={isPredictionCorrect === 'false'} onChange={() => setIsPredictionCorrect('false')} />
                  No
                </label>
              </div>
            </div>

            {isPredictionCorrect === 'true' && (
              window.location.reload()
            )}
            {isPredictionCorrect === 'false' && (
              <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <label>
                  Enter your name:
                  <input type="text" value={userName} onChange={(e) => setUserName(e.target.value)} />
                </label>
                <button onClick={handleFeedback} style={{ marginTop: '10px' }}>
                  Submit Feedback
                </button>
              </div>
            )}
          </div>
        )}
      </Paper>
    </div>
  );
};

export default UserPredict;