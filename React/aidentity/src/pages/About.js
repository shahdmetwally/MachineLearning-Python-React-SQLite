// AboutSection.js
import React, { useState } from "react";
import { Container, Typography, Button, Grid, Paper } from "@mui/material";
import { useSpring, animated } from "react-spring";
import Confetti from "react-confetti";

const AboutSection = () => {
  const [isConfettiActive, setConfettiActive] = useState(false);

  const sectionSpring = useSpring({
    opacity: 1,
    from: { opacity: 0 },
  });

  const paperSpring = useSpring({
    opacity: 1,
    transform: "translate3d(0,0,0)",
    from: { opacity: 0, transform: "translate3d(0,50px,0)" },
  });

  const learnMoreButtonSpring = useSpring({
    opacity: 1,
    transform: "translate3d(0,0,0)",
    from: { opacity: 0, transform: "translate3d(0,50px,0)" },
  });

  const handleConfetti = () => {
    setConfettiActive(true);
    setTimeout(() => {
      setConfettiActive(false);
    }, 5000); // Set a timeout to stop confetti after a few seconds
  };

  return (
    <animated.div
      style={{
        ...sectionSpring,
        minHeight: "90vh",
        display: "flex",
        alignItems: "center",
      }}
    >
      <Container style={{ padding: "20px", textAlign: "center" }}>
        {isConfettiActive && <Confetti />}
        <Paper
          elevation={3}
          style={{
            padding: "40px",
            backgroundColor: "#D9D9D9",
            borderRadius: "10px",
          }}
        >
          <Typography
            variant="h2"
            gutterBottom
            style={{
              fontWeight: "bolder",
              color: "black",
            }}
          >
            A little about AiDentity
          </Typography>
          <Grid container spacing={4} justifyContent="center">
            <Grid item xs={12} sm={6}>
              <animated.div style={paperSpring}>
                <Paper
                  elevation={3}
                  style={{
                    padding: "20px",
                    borderRadius: "8px",
                    height: "20rem",
                    backgroundColor: "#1a353e",
                    color: "white",
                    overflowY: "auto",
                  }}
                >
                  <Typography variant="h4" gutterBottom>
                    Purpose
                  </Typography>
                  <Typography variant="body1">
                    This project's purpose is to make it quick and easy for
                    travelers to get their identity verified using facial
                    recognition.The goal of the system is to be able to detect
                    and identify individuals accurately based on their facial
                    features and structure. This concept addresses possible
                    border control scenarios' purpose being to assist in
                    locations with resource limitations or understaffed and
                    remote locations that would benefit from an automated
                    system. Other cases that would benefit from the use of such
                    a system would be at airports with high passenger volumes.
                  </Typography>
                </Paper>
              </animated.div>
            </Grid>
            <Grid item xs={12} sm={6}>
              <animated.div style={paperSpring}>
                <Paper
                  elevation={3}
                  style={{
                    padding: "20px",
                    borderRadius: "8px",
                    height: "20rem",
                    backgroundColor: "#1a353e",
                    color: "white",
                  }}
                >
                  <Typography variant="h4" gutterBottom>
                    The Internals
                  </Typography>
                  Our AI uses a transfer learned model to identify your images. 
                  When you upload an image it will be stored and used for improving our model. 
                  It is important to disclose that some inaccuracies with the model's  identification of peoples might occur.
                  <Typography variant="body1"></Typography>
                </Paper>
              </animated.div>
            </Grid>
          </Grid>
          <animated.div style={learnMoreButtonSpring}>
            <Button
              variant="contained"
              style={{
                marginTop: "40px",
                background:
                  "linear-gradient(to right, #ff0000, #ff9900, #ff0, #33cc33, #3399ff, #9900cc)",
                borderRadius: "20px",
                color: "#1a353e",
                fontWeight: "bold",
              }}
              size="large"
              onClick={handleConfetti}
            >
              Easter Egg
            </Button>
          </animated.div>
        </Paper>
      </Container>
    </animated.div>
  );
};

export default AboutSection;
