import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Ensure the project root is in the Python path so package imports resolve correctly
# when running this script directly from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modeling.preprocessing.preprocess_halt import preprocess
from modeling.models.halt import HALTModel


class HaltDataset(Dataset):
    """PyTorch Dataset wrapper for HALT training data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # X[idx] shape: (seq_len, features)
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        return x_tensor, y_tensor


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    X_batch, y_batch = zip(*batch)
    lengths = torch.tensor([len(x) for x in X_batch], dtype=torch.long)
    # Pad sequences to the longest in the batch along the sequence dimension
    X_padded = pad_sequence(list(X_batch), batch_first=True, padding_value=0.0)
    y_batch = torch.stack(y_batch)
    return X_padded, lengths, y_batch


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, lengths, y in dataloader:
        X, lengths, y = X.to(device), lengths.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X, lengths)
        loss = criterion(logits.squeeze(-1), y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        preds = (logits.squeeze(-1) > 0).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    return total_loss / total, correct / total


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, lengths, y in dataloader:
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            
            logits = model(X, lengths)
            loss = criterion(logits.squeeze(-1), y)
            
            total_loss += loss.item() * X.size(0)
            preds = (logits.squeeze(-1) > 0).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train HALT model for Uncertainty Quantification")
    parser.add_argument("--dataset", type=str, default="css2-uq/mmlu-traces", help="HuggingFace dataset identifier")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--top_q", type=float, default=0.15)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and preprocessing data...")
    X, y = preprocess(args.dataset)
    
    # Ensure y is 1D for binary classification
    if y.ndim > 1:
        y = y.squeeze(-1)
        
    # Train/Val split (80/20)
    split_idx = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"Dataset sizes -> Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Feature shape: {X_train[0].shape}, Label distribution: {np.mean(y_train):.3f} positive")

    train_dataset = HaltDataset(X_train, y_train)
    val_dataset = HaltDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Initialize model
    input_dim = X_train.shape[-1]
    model = HALTModel(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        top_q=args.top_q
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(args.save_dir, "best_halt_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args)
            }, save_path)
            print(f"  -> Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
