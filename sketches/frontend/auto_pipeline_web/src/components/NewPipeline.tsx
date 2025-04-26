// src/components/NewPipeline.tsx
import React, { useState } from "react";
import {
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemButton,
} from "@mui/material";

const steps = [
  {
    label: "Task Selection",
    options: ["Classification", "Segmentation", "Detection"],
  },
  { label: "Data Path/Type", options: ["Image", "Text", "Audio"] },
  { label: "Model", options: ["ResNet", "BERT", "Transformer"] },
  { label: "Parameters", options: ["Learning Rate", "Batch Size", "Epochs"] },
  { label: "Train/Inference", options: ["Train", "Inference"] },
];

const NewPipeline: React.FC = () => {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [selections, setSelections] = useState<{ [key: number]: string }>({});

  const handleOpenDialog = (index: number) => {
    setSelectedStep(index);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setSelectedStep(null);
  };

  const handleSelectOption = (option: string) => {
    if (selectedStep !== null) {
      setSelections((prev) => ({ ...prev, [selectedStep]: option }));
      handleCloseDialog();
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Sidebar for steps */}
      <div
        style={{
          width: "300px",
          borderRight: "1px solid #ddd",
          padding: "20px",
        }}
      >
        {steps.map((step, index) => (
          <div key={index} style={{ marginBottom: "20px" }}>
            <Button
              variant="outlined"
              onClick={() => handleOpenDialog(index)}
              style={{
                display: "flex",
                justifyContent: "space-between",
                width: "100%",
              }}
            >
              {`STEP ${index + 1}. ${step.label}`}
              {selections[index] && (
                <span style={{ marginLeft: "10px" }}>{selections[index]}</span>
              )}
            </Button>
          </div>
        ))}

        <div style={{ marginTop: "auto", display: "flex", gap: "10px" }}>
          <button style={{ flex: 1 }}>Clear</button>
          <button style={{ flex: 1 }}>Save/Run</button>
        </div>
      </div>

      {/* Visualization area */}
      <div style={{ flex: 1, padding: "20px" }}>
        <h1>Visualization Area</h1>
        <p>Block diagram visualization will go here...</p>
      </div>

      {/* Dialog for selecting options */}
      <Dialog open={dialogOpen} onClose={handleCloseDialog}>
        <DialogTitle>Select an Option</DialogTitle>
        <DialogContent>
          <List>
            {selectedStep !== null &&
              steps[selectedStep].options.map((option) => (
                <ListItem key={option} disablePadding>
                  <ListItemButton onClick={() => handleSelectOption(option)}>
                    {option}
                  </ListItemButton>
                </ListItem>
              ))}
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default NewPipeline;
