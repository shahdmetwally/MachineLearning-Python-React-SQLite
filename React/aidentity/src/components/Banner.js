import React, { useState } from "react";
import { Snackbar, Button } from "@mui/material";
// All done by Sepehr 
const Banner = () => {
  const [open, setOpen] = useState(!localStorage.getItem("disclaimerAccepted"));

  const handleAccept = () => {
    localStorage.setItem("disclaimerAccepted", "true");
    setOpen(false);
  };

  return (
    <Snackbar
      anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      open={open}
      message="Our application will save your images and use them to improve."
      action={
        <Button color="inherit" size="small" onClick={handleAccept}>
          I Understand
        </Button>
      }
    />
  );
};

export default Banner;
