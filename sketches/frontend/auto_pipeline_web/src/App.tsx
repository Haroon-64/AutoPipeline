// src/App.tsx
import React, { useState } from "react";
import Navbar from "./components/Navbar";
import NewPipeline from "./components/NewPipeline";
import Inference from "./components/Inference";

const App: React.FC = () => {
  const [view, setView] = useState("NewPipeline"); // State to toggle between views

  return (
    <div>
      <Navbar setView={setView} />
      <div style={{ padding: "20px" }}>
        {view === "NewPipeline" ? <NewPipeline /> : <Inference />}
      </div>
    </div>
  );
};

export default App;
