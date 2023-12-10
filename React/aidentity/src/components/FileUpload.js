import React, { useState } from "react";
import Input from "@mui/material/Input";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import axios from "axios";
import fileUploadStyle from "./styles/fileUpload.module.css";

const FileUpload = ({ onUpload }) => {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    try {
      const formData = new FormData();
      formData.append("db_file", file);

      const response = await axios.post(
        "http://localhost:8000/retrain",
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
    }
  };

  return (
    <div className={fileUploadStyle.wrapper}>
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
    </div>
  );
};

export default FileUpload;
