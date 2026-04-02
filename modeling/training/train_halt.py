from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from modeling.models.halt import HALTModel
from modeling.preprocessing.preprocess_halt import preprocess
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HaltDataset(Dataset):
    """PyTorch Dataset wrapper for HALT training data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # Compute actual sequence lengths based on non-zero timesteps
        # (preprocess pads with zeros, so this reliably finds the true length)
        self.lengths = torch.tensor(
            [(x != 0).any(dim=-1).sum().item() for x in self.X],
            dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.lengths[idx]

def collate_fn(batch):
    inputs, targets, lengths = zip(*batch)
    # All sequences are already padded to MAX_LEN by preprocess(), so stack works directly
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    lengths = torch.stack(lengths)
    return inputs, targets, lengths

def train_epoch(model, dataloader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (inputs, targets, lengths) in enumerate(dataloader):
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Collect predictions for F1 calculation
        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        
        if batch_idx % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), epoch * len(dataloader) + batch_idx)
    
    # Calculate macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    writer.add_scalar('train/macro_f1', macro_f1, epoch)
    return total_loss / len(dataloader), macro_f1

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Collect predictions for F1 calculation
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(dataloader), macro_f1

def main():
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 4.41e-4
    NUM_EPOCHS = 200
    PATIENCE = 100  # Early stopping patience
    VAL_SPLIT = 0.2

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data (called once)
    X, y = preprocess()

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=42
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"Train correct: {y_train.sum()}, Train incorrect: {len(y_train) - y_train.sum()}")
    print(f"Val correct: {y_val.sum()}, Val incorrect: {len(y_val) - y_val.sum()}")

    # Create datasets and dataloaders
    train_dataset = HaltDataset(X_train, y_train)
    val_dataset = HaltDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model with paper-matching architecture
    model = HALTModel(
        input_dim=25,
        proj_dim=128,
        hidden_size=256,
        num_layers=5,
        dropout=0.4,
        top_q=0.15
    ).to(device)

    # Loss and optimizer (BCEWithLogitsLoss is correct for binary classification with raw logits)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # TensorBoard writer
    writer = SummaryWriter(comment='_halt_training')

    # Early stopping variables
    best_val_f1 = -1.0
    patience_counter = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_halt_model.pth"

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)

        # Validate
        val_loss, val_f1 = eval_epoch(model, val_loader, criterion, device)

        # Log metrics
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/macro_f1', val_f1, epoch)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

        # Update learning rate scheduler
        scheduler.step(val_f1)

        # Early stopping check based on macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved to {best_model_path} with val F1: {val_f1:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Training complete. Best validation macro-F1: {best_val_f1:.4f}')
    writer.close()

if __name__ == '__main__':
    main()
