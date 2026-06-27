"""
train_trajectory_classifier.py — LSTM/1D-CNN trajectory motion classifier.

Trains a deep learning model to classify locus trajectories into 4 motion types:
    0 = Normal diffusion
    1 = Confined diffusion
    2 = Directed motion
    3 = Anomalous subdiffusion

Requires PyTorch. If not available, falls back to sklearn Random Forest baseline.

Usage:
    # Step 1: Generate training data (run this first)
    python3 synthetic_classification_data.py --full

    # Step 2: Train the model
    python3 train_trajectory_classifier.py                # LSTM (default)
    python3 train_trajectory_classifier.py --model cnn    # 1D-CNN
    python3 train_trajectory_classifier.py --model rf     # Random Forest baseline (no PyTorch needed)

    # Step 3: Apply to real data
    python3 train_trajectory_classifier.py --predict
"""

import numpy as np
import os
import sys

# ── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
N_CLASSES = 4
CLASS_NAMES = {0: "normal", 1: "confined", 2: "directed", 3: "subdiffusion"}
MAX_STEPS = 30
DT = 24.5
PIXEL_SIZE_NM = 183.3


# ── Data Loading ────────────────────────────────────────────────────────────

def load_split(split_name: str):
    """Load a data split from .npz files."""
    path = os.path.join(DATA_DIR, f'{split_name}_classification.npz')
    data = np.load(path)
    return data['X'], data['mask'], data['labels'], data['lengths']


def load_features(split_name: str):
    """Load pre-extracted features."""
    path = os.path.join(DATA_DIR, f'{split_name}_features.npz')
    data = np.load(path)
    return data['features'], data['labels']


# ── Random Forest Baseline ──────────────────────────────────────────────────

def train_random_forest():
    """Train and evaluate a Random Forest on hand-crafted features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    print("Loading pre-extracted features...")
    X_train, y_train = load_features('train')
    X_val, y_val = load_features('val')
    X_test, y_test = load_features('test')

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluate
    for name, X, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
        acc = rf.score(X, y)
        print(f"  {name} accuracy: {acc:.4f}")

    y_pred = rf.predict(X_test)
    print(f"\nTest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values())))

    print("Confusion Matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':>15s} {'norm':>8s} {'conf':>8s} {'dir':>8s} {'sub':>8s}")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>15s} {row[0]:8d} {row[1]:8d} {row[2]:8d} {row[3]:8d}")

    # Feature importance
    feat_data = np.load(os.path.join(DATA_DIR, 'train_features.npz'), allow_pickle=True)
    feat_names = list(feat_data['feature_names'])
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop features:")
    for i in sorted_idx[:8]:
        print(f"  {feat_names[i]:>30s}: {importances[i]:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'rf_classifier.joblib')
    joblib.dump(rf, model_path)
    print(f"\nModel saved: {model_path}")

    return rf


# ── PyTorch Models ──────────────────────────────────────────────────────────

def get_lstm_model():
    """Build LSTM classifier for trajectory sequences."""
    import torch
    import torch.nn as nn

    class TrajectoryLSTM(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=64, n_layers=2,
                     n_classes=4, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=True,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x, lengths):
            # Pack padded sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (hidden, _) = self.lstm(packed)

            # Concatenate final hidden states from both directions
            # hidden shape: (n_layers * 2, batch, hidden_dim)
            h_fwd = hidden[-2]  # last layer, forward
            h_bwd = hidden[-1]  # last layer, backward
            h = torch.cat([h_fwd, h_bwd], dim=1)

            return self.classifier(h)

    return TrajectoryLSTM()


def get_cnn_model():
    """Build 1D-CNN classifier for trajectory sequences."""
    import torch
    import torch.nn as nn

    class TrajectoryCNN(nn.Module):
        def __init__(self, input_dim=2, n_classes=4, dropout=0.3):
            super().__init__()
            self.features = nn.Sequential(
                # Conv block 1
                nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),

                # Conv block 2
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),

                # Conv block 3
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # global average pooling
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x, lengths):
            # x: (batch, seq_len, 2) -> (batch, 2, seq_len) for Conv1d
            x = x.transpose(1, 2)

            # Zero out padded positions
            batch_size, channels, max_len = x.shape
            for i in range(batch_size):
                if lengths[i] < max_len:
                    x[i, :, lengths[i]:] = 0

            x = self.features(x)
            x = x.squeeze(-1)  # (batch, 128)
            return self.classifier(x)

    return TrajectoryCNN()


def train_pytorch_model(model_type='lstm'):
    """Train a PyTorch model (LSTM or CNN)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    X_train, mask_train, y_train, len_train = load_split('train')
    X_val, mask_val, y_val, len_val = load_split('val')

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    len_train_t = torch.LongTensor(len_train)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    len_val_t = torch.LongTensor(len_val)

    train_ds = TensorDataset(X_train_t, y_train_t, len_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t, len_val_t)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # Model
    if model_type == 'lstm':
        model = get_lstm_model()
    else:
        model = get_cnn_model()
    model = model.to(device)
    print(f"\nModel: {model_type.upper()}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 50
    best_val_acc = 0
    patience = 10
    patience_counter = 0

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f'{model_type}_classifier.pt')

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch, len_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            len_batch = len_batch.to(device)

            logits = model(X_batch, len_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch, len_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                len_batch = len_batch.to(device)

                logits = model(X_batch, len_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / val_total
        scheduler.step(val_loss_avg)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                  f"val_loss={val_loss_avg:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test set
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    X_test, mask_test, y_test, len_test = load_split('test')
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    len_test_t = torch.LongTensor(len_test).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, len_test_t)
        test_preds = logits.argmax(1).cpu().numpy()

    from sklearn.metrics import classification_report, confusion_matrix
    test_acc = (test_preds == y_test).mean()
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"\nTest Classification Report:")
    print(classification_report(y_test, test_preds, target_names=list(CLASS_NAMES.values())))

    cm = confusion_matrix(y_test, test_preds)
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>15s} {'norm':>8s} {'conf':>8s} {'dir':>8s} {'sub':>8s}")
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>15s} {row[0]:8d} {row[1]:8d} {row[2]:8d} {row[3]:8d}")

    print(f"\nModel saved: {model_path}")
    return model


# ── Predict on Real Data ────────────────────────────────────────────────────

def predict_real_trajectories(model_type='lstm'):
    """Apply the trained classifier to real Oligo-LiveFISH trajectories.

    Uses LSTM/CNN if available, falls back to RF. Saves predictions CSV
    and prints distribution summary for sanity-checking.
    """
    import glob
    import csv
    from synthetic_classification_data import compute_handcrafted_features, pad_sequences

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data for analysis', 'data')

    # Collect all real trajectory CSVs
    print("Scanning for real trajectory CSVs...")
    traj_files = glob.glob(os.path.join(data_root, '**', '*_traj_rela2wholeimg.csv'),
                           recursive=True)
    print(f"  Found {len(traj_files)} trajectory files")

    if not traj_files:
        print("No trajectory files found. Check data path.")
        return

    results = []
    all_displacements = []
    for traj_path in traj_files:
        try:
            traj = np.genfromtxt(traj_path, delimiter=',', skip_header=1)
            if traj.ndim < 2 or len(traj) < 3:
                continue

            x_nm = traj[:, 1] * PIXEL_SIZE_NM
            y_nm = traj[:, 2] * PIXEL_SIZE_NM
            displacements = np.column_stack([np.diff(x_nm), np.diff(y_nm)])

            if len(displacements) < 3:
                continue

            features = compute_handcrafted_features(displacements)
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

            rel_path = os.path.relpath(traj_path, data_root)
            results.append({
                'file': rel_path,
                'n_frames': len(traj),
                'features': features,
            })
            all_displacements.append(displacements)
        except Exception:
            continue

    print(f"  Successfully loaded {len(results)} trajectories")

    # ── Try LSTM/CNN first, fall back to RF ──
    use_dl = False
    dl_path = os.path.join(MODEL_DIR, f'{model_type}_classifier.pt')

    try:
        import torch
        if os.path.exists(dl_path):
            use_dl = True
    except ImportError:
        pass

    if use_dl:
        import torch
        print(f"\nPredicting with {model_type.upper()} model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_type == 'lstm':
            model = get_lstm_model()
        else:
            model = get_cnn_model()
        model.load_state_dict(torch.load(dl_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()

        # Pad displacement sequences
        X_pad, _ = pad_sequences(all_displacements, max_len=MAX_STEPS)
        lengths = np.array([min(len(d), MAX_STEPS) for d in all_displacements])

        X_t = torch.FloatTensor(X_pad).to(device)
        len_t = torch.LongTensor(lengths).to(device)

        with torch.no_grad():
            logits = model(X_t, len_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = logits.argmax(1).cpu().numpy()

        method_label = model_type.upper()
    else:
        # Fall back to Random Forest
        import joblib
        rf_path = os.path.join(MODEL_DIR, 'rf_classifier.joblib')
        if not os.path.exists(rf_path):
            print(f"  No trained model found. Train first.")
            return

        print("\nPredicting with Random Forest...")
        rf = joblib.load(rf_path)
        X = np.array([r['features'] for r in results])
        predictions = rf.predict(X)
        probs = rf.predict_proba(X)
        method_label = "RF"

    # Save predictions
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f'trajectory_predictions_{method_label.lower()}.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'n_frames', 'predicted_class', 'predicted_label',
                         'prob_normal', 'prob_confined', 'prob_directed', 'prob_subdiffusion'])
        for r, pred, prob in zip(results, predictions, probs):
            writer.writerow([
                r['file'], r['n_frames'], int(pred), CLASS_NAMES[int(pred)],
                f"{prob[0]:.3f}", f"{prob[1]:.3f}", f"{prob[2]:.3f}", f"{prob[3]:.3f}"
            ])

    # Summary
    print(f"\n{method_label} Prediction Summary ({len(predictions)} trajectories):")
    for c in range(N_CLASSES):
        n = (predictions == c).sum()
        pct = 100 * n / len(predictions)
        print(f"  {CLASS_NAMES[c]:>15s}: {n:5d} ({pct:5.1f}%)")

    print(f"\nSaved: {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model_type = 'lstm'
    predict_mode = '--predict' in sys.argv

    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_type = sys.argv[idx + 1]

    if predict_mode:
        predict_real_trajectories(model_type=model_type)
    elif model_type == 'rf':
        train_random_forest()
    else:
        try:
            import torch
            train_pytorch_model(model_type)
        except ImportError:
            print("PyTorch not available. Install with: pip install torch")
            print("Falling back to Random Forest baseline...")
            train_random_forest()
