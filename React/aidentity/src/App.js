import React from "react";
import { Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Admin from "./pages/Admin";
import Retrain from "./pages/Retrain";
import ViewModels from "./pages/ViewModels";
import History from "./pages/History";
import About from "./pages/About";
import Banner from "./components/Banner";

function App() {
  return (
    <div>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/history" element={<History />} />
        <Route path="/about" element={<About />} />
        <Route path="/admin" element={<Admin />} />
        <Route path="/retrain" element={<Retrain />} />
        <Route path="/view-models" element={<ViewModels />} />
      </Routes>
      <Banner />
    </div>
  );
}

export default App;
