// src/components/Navbar.tsx
import React from "react";

interface NavbarProps {
  setView: (view: string) => void;
}

const Navbar: React.FC<NavbarProps> = ({ setView }) => {
  return (
    <nav
      style={{
        padding: "10px",
        borderBottom: "1px solid #ddd",
        display: "flex",
        gap: "20px",
      }}
    >
      <button onClick={() => setView("NewPipeline")}>New Pipeline</button>
      <button onClick={() => setView("Inference")}>Inference</button>
      <button>Overview</button>
      <button>Help</button>
    </nav>
  );
};

export default Navbar;
