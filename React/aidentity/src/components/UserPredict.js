import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Grid, Button, Card, CardContent, Paper, Dialog, DialogTitle, DialogActions, DialogContent, TextField } from "@mui/material";
import BadgeIcon from '../static/images/status-icon.png';
//All done by Dimitrios except for code that handles predictions using camera which was done by Shahd
//useEffect code was done by Jennifer
const UserPredict = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [predictionWidth, setPredictionWidth] = useState("400px");
  const [predictionHeight, setPredictionHeight] = useState("490px");
  const [blobForFeedback, setBlobForFeedback] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const fileInputRef = useRef(null);
  const [isPredictionCorrect, setIsPredictionCorrect] = useState(null);
  const [userName, setUserName] = useState("");
  const [videoStream, setVideoStream] = useState(null);
  const [boundingBox, setBoundingBox] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    const previewURL = URL.createObjectURL(file);
    setImagePreview(previewURL);
    setPrediction(null);
    setBoundingBox(null);
    setIsPredictionCorrect(null);
    setUserName("");
  };

  useEffect(() => {
    if (imagePreview) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const image = new Image();
      image.src = imagePreview;
      image.onload = () => {
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);

        if (boundingBox) {
          ctx.strokeStyle = "red";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.rect(
            boundingBox[0],
            boundingBox[1],
            boundingBox[2],
            boundingBox[3]
          );
          ctx.stroke();
        }
      };
    }
  }, [imagePreview, boundingBox]);
  const handlePredictFile = () => {
    setBoundingBox(null);
    const formData = new FormData();
    formData.append("image", selectedFile);

    axios
      .post(process.env.REACT_APP_SERVER_ENDPOINT + "/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((response) => {
        console.log(response);
        setPrediction(response.data.score);
        setBoundingBox(response.data.box);
        // Save the blob directly in the state for feedback
        response.data.blob && setBlobForFeedback(response.data.blob);
      })
      .catch((error) => {
        console.error("Error predicting:", error);
      });
  };

  const handlePredictCamera = async () => {
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      const timestamp = new Date().getTime();
      const fileName = `snapshot_${timestamp}.jpg`;
      const formData = new FormData();
      formData.append("image", blob, fileName);

      try {
        const response = await axios.post(
          process.env.REACT_APP_SERVER_ENDPOINT + "/predict",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );
        console.log(response);
        setPrediction(response.data.score);
        // Save the blob directly in the state for feedback
        response.data.blob && setBlobForFeedback(response.data.blob);
        setPredictionWidth("auto");
        setPredictionHeight("auto");
      } catch (error) {
        console.error("Error predicting:", error);
      }
    }, "image/jpeg");
  };

  const handleFeedback = () => {
    let imageForFeedback;

    if (isPredictionCorrect !== null) {
      const formData = new FormData();

      if (videoStream) {
        // Camera scenario
        canvasRef.current.toBlob((blob) => {
          formData.append("image", blob);
          imageForFeedback = blob;
          formData.append("is_correct", isPredictionCorrect);

          if (isPredictionCorrect === "false" && userName.trim() !== "") {
            formData.append("user_name", userName.trim());
          }

          sendFeedback(formData, imageForFeedback);
        }, "image/jpeg");
      } else if (selectedFile) {
        // File scenario
        formData.append("image", selectedFile);
        imageForFeedback = selectedFile;
        formData.append("is_correct", isPredictionCorrect);

        if (isPredictionCorrect === "false" && userName.trim() !== "") {
          formData.append("user_name", userName.trim());
        }

        sendFeedback(formData, imageForFeedback);
        setPredictionWidth("auto");
        setPredictionHeight("auto");
      }
    }
  };

  const sendFeedback = (formData, imageForFeedback) => {
    axios
      .post(process.env.REACT_APP_SERVER_ENDPOINT + "/feedback", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        })
      .then((response) => {
        console.log(response);
        alert("Feedback submitted!");
        setTimeout(() => {
          window.location.reload();
        }, 1000);
      })
      .catch((error) => {
      console.error("Error submitting feedback:", error);
    });
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
      console.error("Error accessing camera:", error);
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
  const handleClosePopup = () => {
    // Close the dialog
    setTimeout(() => {
      window.location.reload();
    }, 1000);
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-around",
        alignItems: "center",
      }}
    >
      <Grid container spacing={9} justify="center" alignItems="center">
        <Grid item>
          <Paper
            elevation={3}
            style={{
              padding: "20px",
              top: "100px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              backgroundColor: "#D9D9D9",
            }}
          >
            <h2>Make A Prediction</h2>

            {/* Option 1: Upload File */}
            {(!videoStream || selectedFile) && (
              <label htmlFor="file-upload" style={{ marginBottom: "20px" }}>
                <Button
                  variant="contained"
                  style={{
                    backgroundColor: "#1A353E",
                    color: "white",
                    marginTop: "20px",
                  }}
                  onClick={triggerFileInput}
                >
                  Choose file
                </Button>
                <input
                  type="file"
                  id="file-upload"
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                  ref={fileInputRef}
                />
              </label>
            )}
            <canvas
              ref={canvasRef}
              style={{ maxWidth: "100%", marginTop: "10px" }}
            />
            {selectedFile && (
              <Button
                variant="contained"
                onClick={handlePredictFile}
                style={{
                  backgroundColor: "#1A353E",
                  color: "white",
                  marginTop: "20px",
                }}
              >
                Predict from File
              </Button>
            )}

            {/* Option 2: Use Camera */}
            {selectedFile ? null : (
              <>
                <video
                  ref={videoRef}
                  style={{
                    maxWidth: "300px",
                    marginTop: "0px",
                    display: videoStream ? "block" : "none",
                  }}
                  autoPlay
                ></video>
                <Button
                  variant="contained"
                  onClick={toggleCamera}
                  style={{
                    backgroundColor: "#1A353E",
                    color: "white",
                    marginTop: "50px",
                  }}
                >
                  {videoStream ? "Stop Camera" : "Start Camera"}
                </Button>
                {videoStream && (
                  <Button
                    variant="contained"
                    onClick={handlePredictCamera}
                    style={{
                      backgroundColor: "#1A353E",
                      color: "white",
                      marginTop: "50px",
                    }}
                  >
                    Predict from Camera
                  </Button>
                )}
              </>
            )}
          </Paper>
        </Grid>
        <Grid item>
          <Paper
            className="prediction-column"
            elevation={3}
            style={{
              padding: "20px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              textAlign: "center",
              backgroundColor: "#D9D9D9",
              width: predictionWidth,
              height: predictionHeight,
            }}
          >
          <img src={BadgeIcon} alt="status" style={{ width: 80, height: 80, borderRadius: '50%', color: "#1A353E" }} />
            {prediction && (
              <div style={{ alignItems: "center" }}>
                <Card style={{ marginBottom: "10px", width: "300px", borderRadius: "10px"}}>
                  <CardContent>
                    <div style={{ backgroundColor: "white", padding: "10px" }}>
                      <p style={{fontWeight: "bold"}}>Name:</p> {prediction}
                    </div>
                  </CardContent>
                </Card>
                <Card style={{ width: "300px", borderRadius: "10px"}}>
                  <CardContent>
                    <div style={{ backgroundColor: "white" }}>
                      <p style={{fontWeight: "bold"}}>Status:{" "}</p>
                      <span style={{color: prediction === "Unknown" ? "red" : "green" }}>
                        {prediction === "Unknown" ? "Access denied " : "Access granted "}
                        <div>
                        {prediction === "Unknown" ? (
                            <i className="fas fa-times" style={{ fontSize: '60px' }}></i>
                          ) : (
                            <i className="fas fa-check" style={{ fontSize: '60px' }}></i>
                          )}
                        </div>
                      </span>
                    </div>
                  </CardContent>
                </Card>

                <div
                  style={{
                    marginTop: "20px",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                  }}
                >
                  <label>Is the person recognized correctly?</label>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginTop: "10px",
                    }}
                  >
                    <label>
                      <input
                        type="radio"
                        value="true"
                        checked={isPredictionCorrect === "true"}
                        onChange={() => setIsPredictionCorrect("true")}
                      />
                      Yes
                    </label>
                    <label>
                      <input
                        type="radio"
                        value="false"
                        checked={isPredictionCorrect === "false"}
                        onChange={() => setIsPredictionCorrect("false")}
                      />
                      No
                    </label>
                  </div>
                </div>

                {isPredictionCorrect === "true" && window.location.reload()}
                {isPredictionCorrect === "false" && (
                  <div>
                <Dialog open={true} onClose={handleClosePopup}>
                  <DialogTitle>Feedback Submitted</DialogTitle>
                  <DialogContent>
                  <label>
                    Enter the correct name:
                    <TextField
                      type="text"
                      value={userName}
                      onChange={(e) => setUserName(e.target.value)}
                    />
                  </label>
                  </DialogContent>
                  <DialogActions>
                    <Button onClick={handleClosePopup} color="primary">
                      Close
                    </Button>
                    <Button onClick={handleFeedback} color="primary">
                      Submit
                    </Button>
                  </DialogActions>
                </Dialog>
                </div>
                )}
                </div>
            )}
          </Paper>
        </Grid>
      </Grid>
    </div>
  );
};

export default UserPredict;
