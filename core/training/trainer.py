import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from livelossplot import PlotLosses
from tqdm import tqdm

def trainer(model, epochs, train_loader, val_loader, optimizer, criterion, device):
    '''
    Takes the model to be trained, number of epochs, training DataLoader, validation DataLoader, 
    optimizer, loss criterion, and device, and returns the trained model.
    '''
    model.to(device)
    liveloss = PlotLosses()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=0.0001)

    for epoch in range(epochs):
        logs = {}
        
        # Training phase
        model.train()
        train_loss, correct = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = correct / len(train_loader.dataset)
        logs['loss'] = train_loss
        logs['accuracy'] = train_accuracy

        # Validation phase
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)
        logs['val_loss'] = val_loss
        logs['val_accuracy'] = val_accuracy

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Plot training and validation metrics
        liveloss.update(logs)
        liveloss.send()
    
    return model




    

    
