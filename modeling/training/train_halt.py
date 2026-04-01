import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from modeling.models.halt import HALTModel
from modeling.preprocessing.preprocess_halt import preprocess
from torch.utils.tensorboard import SummaryWriter

class HaltDataset(Dataset):
    """PyTorch Dataset wrapper for HALT training data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = torch.stack(targets)
    return inputs, targets, lengths

def train_epoch(model, dataloader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets, lengths) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + batch_idx)
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    PATIENCE = 10  # Early stopping patience

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    X_train, y_train = preprocess()
    X_val, y_val = preprocess()  # Assuming separate validation data

    # Create datasets and dataloaders
    train_dataset = HaltDataset(X_train, y_train)
    val_dataset = HaltDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model with enhanced architecture
    model = HALTModel(
        input_dim=25,
        proj_dim=256,      # Increased projection dimension
        hidden_size=512,   # Increased hidden size
        num_layers=8,      # More layers for better representation
        dropout=0.3,       # Reduced dropout
        top_q=0.1          # More aggressive top-q sampling
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # TensorBoard writer
    writer = SummaryWriter(comment='_halt_training')

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = 'best_halt_model.pth'

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)

        # Validate
        val_loss = eval_epoch(model, val_loader, criterion, device)

        # Log metrics
        writer.add_scalar('val/loss', val_loss, epoch)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with val loss: {val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Training complete. Best validation loss: {best_val_loss:.4f}')
    writer.close()

if __name__ == '__main__':
    main()
