"""
PyTorch Neural Network for Fraud Detection
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import setup_logger

logger = setup_logger("pytorch_model")


class FraudDataset(Dataset):
    """PyTorch Dataset for fraud detection"""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Args:
            X: Features DataFrame
            y: Target Series
        """
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FraudDetectionNN(nn.Module):
    """
    Fully Connected Neural Network for Fraud Detection
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.3
    ):
        """
        Initialize neural network

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(FraudDetectionNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        logger.info(f"Initialized NN: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")

    def forward(self, x):
        return self.model(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class PyTorchFraudDetector:
    """PyTorch-based fraud detection model"""

    def __init__(
        self,
        hidden_dims: list = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        device: str = None
    ):
        """
        Initialize PyTorch model

        Args:
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            focal_alpha: Focal loss alpha
            focal_gamma: Focal loss gamma
            device: Device to use ('cpu' or 'cuda')
        """
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.model = None
        self.feature_names = None
        self.best_threshold = 0.5
        self.training_history = {'train_loss': [], 'val_loss': []}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        """
        Train the neural network

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("="*70)
        logger.info("Training PyTorch Neural Network")
        logger.info("="*70)

        # Store feature names
        self.feature_names = X_train.columns.tolist()
        input_dim = len(self.feature_names)

        # Initialize model
        self.model = FraudDetectionNN(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Loss and optimizer
        criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create datasets
        train_dataset = FraudDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = FraudDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        # Training loop
        logger.info(f"Training for {self.epochs} epochs...")

        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).view(-1, 1)

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.training_history['train_loss'].append(train_loss)

            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device).view(-1, 1)

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self.training_history['val_loss'].append(val_loss)

                # Log progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.epochs}] "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f}"
                    )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.epochs}] Train Loss: {train_loss:.4f}")

        logger.info("✓ Training completed")

        return self

    def calibrate_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_metric: str = 'f1'
    ):
        """
        Find optimal classification threshold

        Args:
            X_val: Validation features
            y_val: Validation labels
            target_metric: Metric to optimize
        """
        logger.info(f"Calibrating threshold to optimize {target_metric}...")

        y_pred_proba = self.predict_proba(X_val)

        best_score = 0
        best_threshold = 0.5

        thresholds = np.arange(0.1, 0.9, 0.05)

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if target_metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif target_metric == 'precision':
                score = precision_score(y_val, y_pred, zero_division=0)
            elif target_metric == 'recall':
                score = recall_score(y_val, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.best_threshold = best_threshold

        logger.info(f"Best threshold: {best_threshold:.3f}")
        logger.info(f"Best {target_metric}: {best_score:.4f}")

        return best_threshold

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Predict fraud labels"""
        if threshold is None:
            threshold = self.best_threshold

        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= threshold).astype(int)

        return y_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities"""
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()

        dataset = FraudDataset(X, pd.Series([0] * len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions).flatten()

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance"""
        logger.info("="*70)
        logger.info("Model Evaluation")
        logger.info("="*70)

        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'threshold': self.best_threshold
        }

        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f} (Target: > 0.85)")
        logger.info(f"Precision: {metrics['precision']:.4f} (Target: > 0.80)")
        logger.info(f"Recall:    {metrics['recall']:.4f} (Target: > 0.75)")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f} (Target: > 0.78)")

        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
        logger.info(f"FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")

        return metrics

    def save(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = path.parent / f"{path.stem}_model.pth"
        torch.save(self.model.state_dict(), model_path)

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'best_threshold': self.best_threshold,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'input_dim': len(self.feature_names)
        }

        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Saved model to {model_path}")
        logger.info(f"✓ Saved metadata to {metadata_path}")

    def load(self, path: str):
        """Load model from disk"""
        path = Path(path)

        # Load metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Initialize model
        self.feature_names = metadata['feature_names']
        self.best_threshold = metadata['best_threshold']
        self.hidden_dims = metadata['hidden_dims']

        self.model = FraudDetectionNN(
            input_dim=metadata['input_dim'],
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Load weights
        model_path = path.parent / f"{path.stem}_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        logger.info(f"✓ Loaded model from {model_path}")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "processed"

    logger.info("Loading processed data...")
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")['Class']
    y_test = pd.read_parquet(data_dir / "y_test.parquet")['Class']

    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Train model
    model = PyTorchFraudDetector(
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=256,
        epochs=50
    )

    model.train(X_train_split, y_train_split, X_val, y_val)
    model.calibrate_threshold(X_val, y_val, target_metric='f1')
    metrics = model.evaluate(X_test, y_test)

    # Save model
    model_path = project_root / "models" / "pytorch_fraud"
    model.save(str(model_path))

    logger.info("\n✓ PyTorch training completed!")
