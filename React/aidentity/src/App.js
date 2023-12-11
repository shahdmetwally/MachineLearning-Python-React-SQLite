import React from "react";
import { Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Admin from "./pages/Admin";
import Retrain from "./pages/Retrain";
import ViewModels from "./pages/ViewModels";

function App() {
  return (
    <div>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/admin" element={<Admin />} />
        <Route path="/retrain" element={<Retrain />} />
        <Route path="/view-models" element={<ViewModels />} />
      </Routes>
    </div>
  );
}

export default App;
