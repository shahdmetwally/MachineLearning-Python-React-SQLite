import React, { useState, useEffect } from "react";
import { Typography, Grid } from "@mui/material";
import axios from "axios";
import Table from "../components/Table";

const ViewModels = () => {
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);

  useEffect(() => {
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

    fetchModels();
  }, []);

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
      </Grid>
    </Grid>
  );
};

export default ViewModels;
