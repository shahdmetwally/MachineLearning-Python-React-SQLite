import React, { useState } from "react";
import { Snackbar, Button } from "@mui/material";

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
      message="This website may gain consciousness and we're not liable for mischievous behaviour."
      action={
        <Button color="inherit" size="small" onClick={handleAccept}>
          I Understand
        </Button>
      }
    />
  );
};

export default Banner;
