// src/components/Inference.tsx
import React from "react";

const Inference: React.FC = () => {
  return (
    <div style={{ padding: "20px" }}>
      {/* Header */}
      <h1>Inference - Scene 2</h1>

      {/* Tabs */}
      <div
        style={{
          borderBottom: "1px solid #ddd",
          paddingBottom: "10px",
          display: "flex",
          gap: "10px",
        }}
      >
        <button>Load Model</button>
        <button>Task Selection</button>
        <button>File (if not generation)</button>
        <button>Prompt (if generation)</button>
      </div>

      {/* Path Inputs */}
      <div style={{ marginTop: "20px", display: "flex", gap: "20px" }}>
        <div>
          <label>Path:</label>
          <input
            type="text"
            style={{ display: "block", marginTop: "5px", width: "200px" }}
            placeholder="Path"
          />
        </div>
        <div>
          <label>Path:</label>
          <input
            type="text"
            style={{ display: "block", marginTop: "5px", width: "200px" }}
            placeholder="Path"
          />
        </div>
      </div>

      {/* Main Content Area */}
      <div
        style={{ border: "1px solid #ddd", marginTop: "20px", height: "400px" }}
      >
        {/* Large visualization or result area */}
      </div>

      {/* Bottom Scroll Indicator (for visual representation) */}
      <div
        style={{
          marginTop: "20px",
          height: "20px",
          backgroundColor: "#f0f0f0",
        }}
      >
        {/* Scroll indicator - represents progress */}
        <div
          style={{ width: "30%", height: "100%", backgroundColor: "#888" }}
        ></div>
      </div>
    </div>
  );
};

export default Inference;
