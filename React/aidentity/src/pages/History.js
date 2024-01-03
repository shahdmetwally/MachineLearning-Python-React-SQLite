import React, { useState, useEffect } from "react";
import axios from "axios";
import { Paper } from "@mui/material";

function History() {
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/predictions");
        setPredictions(response.data);
      } catch (error) {
        console.error("Error while fetching predictions:", error);
      }
    };

    fetchPredictions();
  }, []);

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "90vh",
      }}
    >
      <Paper
        elevation={3}
        style={{
          padding: "20px",
          backgroundColor: "#1A353E",
          width: "95%",
          borderRadius: "20px",
        }}
      >
        <h2
          style={{
            color: "#ffff",
            marginBottom: "10px",
            textAlign: "center",
          }}
        >
          History
        </h2>
        <Paper
          elevation={3}
          style={{
            height: "60vh",
            overflowY: "auto",
            padding: "40px",
            backgroundColor: "#D9D9D9",
            borderRadius: "10px",
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
          }}
        >
          {predictions.map((prediction) => (
            <div
              key={prediction.id}
              style={{
                display: "flex",
                height: "90px",
                backgroundColor: "white",
                borderRadius: "8px",
                overflow: "hidden",
                width: "30%",
              }}
            >
              <div style={{ height: "70px", width: "70px" }}>
                <img
                  src={`data:image/jpeg;base64,${prediction.image}`}
                  alt="Prediction"
                  style={{
                    width: "90px",
                    height: "90px",
                    objectFit: "cover",
                  }}
                />
              </div>
              <div
                style={{
                  paddingLeft: "40px",
                  paddingTop: "20px",
                  fontSize: "small",
                }}
              >
                <p>Prediction: {prediction.score}</p>
                <p>Created At: {prediction.created_at}</p>
              </div>
            </div>
          ))}
        </Paper>
      </Paper>
    </div>
  );
}

export default History;
