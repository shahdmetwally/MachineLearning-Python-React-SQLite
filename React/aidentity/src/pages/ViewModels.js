import React, { useState, useEffect } from "react";
import { Typography } from "@mui/material";
import axios from "axios";

const ModelsPage = () => {
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
    <div>
      <Typography variant="h4" gutterBottom>
        Available Models
      </Typography>

      {models.length > 0 ? (
        <div>
          <Typography variant="h6">Models:</Typography>
          <ul>
            {models.map((model) => (
              <li key={model}>{model}</li>
            ))}
          </ul>

          <Typography variant="h6">Active Model:</Typography>
          <Typography>{activeModel}</Typography>
        </div>
      ) : (
        <Typography>No models available.</Typography>
      )}
    </div>
  );
};

export default ModelsPage;
