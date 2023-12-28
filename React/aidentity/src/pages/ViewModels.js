import React, { useState, useEffect } from "react";
import { Typography, Grid, TextField, Button } from "@mui/material";
import axios from "axios";
import Table from "../components/Table";

const ViewModels = () => {
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const [versionInput, setVersionInput] = useState("");

  const fetchModels = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/models");
      const { models, active_model: activeModel } = response.data;
      setModels(models);
      setActiveModel(activeModel);
    } catch (error) {
      console.error("Error fetching models:", error);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleSetActiveModel = async (version) => {
    const formData = new FormData();
    formData.append("version", versionInput);
    try {
      const response = await axios.put(
        "http://127.0.0.1:8000/model",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("Set Active Model Response:", response.data);
      fetchModels();
    } catch (error) {
      console.error("Error setting active model:", error);
    }
  };

  return (
    <Grid container spacing={2} justifyContent="center">
      <Grid item xs={12} sm={8} md={6} lg={8}>
        <Typography
          sx={{
            paddingTop: "2rem",
            fontWeight: "bold",
          }}
          variant="h4"
          gutterBottom
        >
          Available Models
        </Typography>

        {models.length > 0 ? (
          <div>
            <Table models={models} headerText="All Models" />
            <Table models={[activeModel]} headerText="Active Model" />
          </div>
        ) : (
          <Typography>No models available.</Typography>
        )}

        {/* Form for setting the active model */}
        <Typography
          sx={{
            paddingTop: "2rem",
            fontWeight: "bold",
          }}
          variant="h5"
          gutterBottom
        >
          Set Active Model
        </Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <TextField
              label="Version"
              variant="outlined"
              value={versionInput}
              onChange={(e) => setVersionInput(e.target.value)}
            />
            <Typography
              sx={{
                fontSize: "12px",
                color: "black",
                marginTop: "8px",
              }}
            >
              Example: 20231204215715
            </Typography>
          </Grid>
          <Grid>
            <Button
              variant="contained"
              color="primary"
              onClick={handleSetActiveModel}
              sx={{
                backgroundColor: "#1a353e",
                borderRadius: "10px",
                color: "#fff",
                marginLeft: "10px",
                marginBottom: "8px",
              }}
            >
              Set
            </Button>
          </Grid>
        </Grid>
      </Grid>
    </Grid>
  );
};

export default ViewModels;
