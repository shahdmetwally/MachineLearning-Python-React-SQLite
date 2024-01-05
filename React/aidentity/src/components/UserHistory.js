import React, { useState, useEffect } from "react";
import '@fortawesome/fontawesome-free/css/all.css';
import axios from "axios";
import { Paper } from "@mui/material";
//All done by Dimitrios
//Part of the design for the prediction history was done by Jennifer
const UserHistory = () => {
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await axios.get(
          process.env.REACT_APP_SERVER_ENDPOINT + "/predictions"
        );
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
        maxWidth: "1000px",
      }}
    >
      <Paper
        elevation={3}
        style={{ padding: "20px", backgroundColor: "#1A353E", width: "80%" }}
      >
        <h2
          style={{
            color: "white",
            marginBottom: "10px",
            textAlign: "center",
          }}
        >
          History
        </h2>
        <Paper
          elevation={3}
          style={{
            maxHeight: "400px",
            overflowY: "auto",
            padding: "20px",
            backgroundColor: "#D9D9D9",
          }}
        >
          <ul style={{ listStyle: "none", padding: 0 }}>
            {predictions.map((prediction) => (
              <li key={prediction.id} style={{ marginBottom: "20px" }}>
                <div
                  style={{
                    display: "flex",
                    height: "70px",
                    backgroundColor: "white",
                    borderRadius: "8px",
                    overflow: "hidden",
                    paddingRight: "150px",
                    position: "relative", 
                  }}
                >
                  <div style={{ height: "70px", width: "70px" }}>
                    <img
                      src={`data:image/jpeg;base64,${prediction.image}`}
                      alt="Prediction"
                      style={{
                        width: "70px",
                        height: "70px",
                        objectFit: "cover",
                      }}
                    />
                  </div>
                  <div
                    style={{
                      marginLeft: "10px",
                      display: "flex",
                      alignItems: "center",
                    }}
                  >
                    <div
                      style={{
                        padding: "10px",
                        fontSize: "small",
                        display: "flex",
                        flexDirection: "row",
                        alignItems: "center",
                        marginRight: "10px",
                      }}
                    >
                      <p style={{ whiteSpace: 'nowrap', paddingRight: "10px" }}>
                        {prediction.score}
                      </p>
                      <div
                      style={{
                        margin: "135px",
                        width: "40px",
                        height: "40px",
                        borderRadius: "50%",
                        backgroundColor: prediction.score === "Unknown" ? "red" : "green",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "white",
                        position: "absolute", 
                      }}
                    >
                      {prediction.score === "Unknown" ? (
                        <i className="fas fa-times"></i>
                      ) : (
                        <i className="fas fa-check"></i>
                      )}
                    </div>
                    </div>                  
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </Paper>
      </Paper>
    </div>
  );
};

export default UserHistory;
