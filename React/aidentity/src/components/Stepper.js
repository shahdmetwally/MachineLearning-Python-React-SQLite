import React, { useState, useEffect } from "react";
import { styled } from "@mui/material/styles";
import Stack from "@mui/material/Stack";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import VideoLabelIcon from "@mui/icons-material/VideoLabel";
import Typography from "@mui/material/Typography";
import StepConnector, {
  stepConnectorClasses,
} from "@mui/material/StepConnector";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import FileUpload from "./FileUpload";
import {
  Table,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  TableHead,
} from "@mui/material";
import Grid from "@mui/material/Grid";
import axios from "axios";

const ColorlibConnector = styled(StepConnector)(({ theme }) => ({
  [`&.${stepConnectorClasses.alternativeLabel}`]: {
    top: 22,
  },
  [`&.${stepConnectorClasses.active}`]: {
    [`& .${stepConnectorClasses.line}`]: {
      backgroundImage: "linear-gradient( 95deg, #6e99a7 0%, #1a353e 100%)",
    },
  },
  [`&.${stepConnectorClasses.completed}`]: {
    [`& .${stepConnectorClasses.line}`]: {
      backgroundImage: "linear-gradient( 95deg, #6e99a7 0%, #1a353e 100%)",
    },
  },
  [`& .${stepConnectorClasses.line}`]: {
    height: 3,
    border: 0,
    backgroundColor:
      theme.palette.mode === "dark" ? theme.palette.grey[800] : "#eaeaf0",
    borderRadius: 1,
  },
}));

const ColorlibStepIconRoot = styled("div")(({ theme, ownerState }) => ({
  backgroundColor:
    theme.palette.mode === "dark" ? theme.palette.grey[700] : "#ccc",
  zIndex: 1,
  color: "#fff",
  width: 50,
  height: 50,
  display: "flex",
  borderRadius: "50%",
  justifyContent: "center",
  alignItems: "center",
  ...(ownerState.active && {
    backgroundImage: "linear-gradient( 136deg, #6e99a7 0%, #1a353e 100%)",
    boxShadow: "0 4px 10px 0 rgba(0,0,0,.25)",
  }),
  ...(ownerState.completed && {
    backgroundImage: "linear-gradient( 136deg, #6e99a7 0%, #1a353e 100%)",
  }),
}));

function ColorlibStepIcon(props) {
  const { active, completed, className } = props;

  const icons = {
    1: <CloudUploadIcon />,
    2: <VideoLabelIcon />,
  };

  return (
    <ColorlibStepIconRoot
      ownerState={{ completed, active }}
      className={className}
    >
      {icons[String(props.icon)]}
    </ColorlibStepIconRoot>
  );
}

const steps = ["Upload new data", "View Metrics"];

const CustomizedSteppers = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [activeModel, setActiveModel] = useState(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get(
          process.env.REACT_APP_SERVER_ENDPOINT + "/models"
        );
        const { active_model: fetchedActiveModel } = response.data;
        setActiveModel(fetchedActiveModel);
      } catch (error) {
        console.error("Error fetching active model:", error);
      }
    };

    fetchModels();
  }, []);

  useEffect(() => {
    console.log("Active Model Changed:", activeModel);
  }, [activeModel]);

  const handleUpload = (uploadedMetrics) => {
    setMetrics(uploadedMetrics);
    setActiveStep(1);
  };

  const handleRestart = () => {
    setMetrics(null);
    setActiveStep(0);
  };

  return (
    <Stack sx={{ width: "100%", marginTop: "2rem" }} spacing={4}>
      <Typography
        variant="h5"
        sx={{ fontWeight: "bold", textAlign: "center", padding: "1rem" }}
      >
        Current Active Model: {activeModel}
      </Typography>
      <Stepper
        alternativeLabel
        activeStep={activeStep}
        connector={<ColorlibConnector />}
      >
        {steps.map((label, index) => (
          <Step key={label}>
            <StepLabel StepIconComponent={ColorlibStepIcon}>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      {activeStep === 0 && <FileUpload onUpload={handleUpload} />}
      {activeStep === 1 && (
        <Grid container spacing={2} justifyContent="center">
          <Grid item xs={12} sm={8} md={6} lg={7}>
            <Typography
              variant="h6"
              sx={{ fontWeight: "bold", textDecoration: "underline" }}
            >
              Retrained Metrics:
            </Typography>
            <Paper sx={{ padding: 0, marginBottom: 2, marginTop: 2 }}>
              <TableContainer>
                <Table>
                  <TableHead sx={{ backgroundColor: "#1a353e" }}>
                    <TableRow>
                      <TableCell
                        sx={{
                          fontWeight: "bold",
                          textDecoration: "underline",
                          color: "white",
                        }}
                      >
                        Metric
                      </TableCell>
                      <TableCell
                        align="right"
                        sx={{
                          fontWeight: "bold",
                          textDecoration: "underline",
                          color: "white",
                        }}
                      >
                        Value
                      </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(metrics).map(([key, value]) => (
                      <TableRow key={key}>
                        <TableCell align="left">
                          {key.replace(/_/g, " ")}
                        </TableCell>
                        <TableCell align="right">{value}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
            <Button
              sx={{
                backgroundColor: "#1a353e",
                borderRadius: "20px",
                color: "#fff",
              }}
              variant="contained"
              color="primary"
              onClick={handleRestart}
            >
              Restart
            </Button>
          </Grid>
        </Grid>
      )}
    </Stack>
  );
};

export default CustomizedSteppers;
