import React, { useState } from "react";
import Input from "@mui/material/Input";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import LinearProgress from "@mui/material/LinearProgress";
import axios from "axios";
import fileUploadStyle from "./styles/fileUpload.module.css";
// All done by Sepehr 
const FileUpload = ({ onUpload }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("db_file", file);

      const response = await axios.post(
        process.env.REACT_APP_SERVER_ENDPOINT + "/retrain",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      onUpload(response.data);
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={fileUploadStyle.wrapper}>
      <div style={{ display: "flex", flexDirection: "column" }}>
        <Paper
          sx={{
            width: 384,
            height: 160,
            backgroundColor: "#d9d9d9",
            padding: "2rem",
          }}
          elevation={24}
        >
          <Input
            className={fileUploadStyle.file}
            type="file"
            onChange={handleFileChange}
            sx={{ mb: 2, display: "block" }}
          />
          <Button
            sx={{
              backgroundColor: "#1a353e",
              borderRadius: "20px",
              color: "#fff",
            }}
            variant="contained"
            color="primary"
            onClick={handleUpload}
          >
            Upload and View Metrics
          </Button>
        </Paper>
        {loading && (
          <Box
            sx={{
              width: "100%",
              marginTop: "3rem",
            }}
          >
            <LinearProgress />
          </Box>
        )}
      </div>
    </div>
  );
};

export default FileUpload;

//xotwod
