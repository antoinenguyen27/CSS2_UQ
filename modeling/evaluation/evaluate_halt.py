import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from modeling.models.halt import HALTModel
from modeling.preprocessing.preprocess_halt import preprocess, MAX_LEN, FEATURE_DIM

class HaltDataset(Dataset):
    """PyTorch Dataset wrapper for HALT evaluation data."""
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

def brier_score(y_true, y_pred):
    """Compute Brier score: mean((y_true - y_pred)^2)"""
    return torch.mean((y_true - y_pred) ** 2)

def evaluate_model(model, dataloader, device):
    model.eval()
    brier_scores = []
    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            probs = model.predict_proba(inputs, lengths)
            brier = brier_score(targets, probs)
            brier_scores.append(brier.item())
    return np.mean(brier_scores)

def main():
    # Evaluation configuration
    BATCH_SIZE = 32
    MODEL_PATH = 'best_halt_model.pth'  # Path to trained model checkpoint

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess test data
    X_test, y_test = preprocess()  # Assuming separate test data

    # Create dataset and dataloader
    test_dataset = HaltDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = HALTModel(
        input_dim=25,
        proj_dim=256,
        hidden_size=512,
        num_layers=8,
        dropout=0.3,
        top_q=0.1
    ).to(device)

    # Load trained model checkpoint
    if Path(MODEL_PATH).exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model checkpoint from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model checkpoint {MODEL_PATH} not found")

    # Evaluate model
    brier = evaluate_model(model, test_loader, device)
    print(f"Evaluation complete. Brier score: {brier:.4f}")

if __name__ == '__main__':
    main()
