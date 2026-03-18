"""
BIS (Bispectral Index) Prediction from ECG and PPG Signals
===========================================================
- Reads .vital files, extracts ECG/PPG waveforms and BIS labels
- Creates 20-second sliding windows
- Classifies BIS into 4 depth-of-anesthesia categories:
    0: Deep Anesthesia    (BIS 0-40)
    1: Moderate Anesthesia (BIS 40-60)
    2: Light Anesthesia    (BIS 60-80)
    3: Awake              (BIS 80-100)
- Trains a 1D CNN + Transformer model (PyTorch)
- Also trains XGBoost on handcrafted features for comparison
"""

import os
import glob
import numpy as np
import pandas as pd
import warnings
import pickle
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# 1. DATA EXTRACTION & SEGMENTATION
# ──────────────────────────────────────────────────────────

SAMPLE_RATE = 500        # Hz for waveforms
WINDOW_SEC = 20          # seconds per segment
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC  # 10000 samples
STRIDE_SEC = 10          # sliding window stride (50% overlap)
STRIDE_SAMPLES = SAMPLE_RATE * STRIDE_SEC
MIN_SQI = 50             # minimum Signal Quality Index to use BIS value
DATA_DIR = "DoA data"
OUTPUT_DIR = "processed_data"

# BIS category mapping
def bis_to_category(bis_value):
    if bis_value < 40:
        return 0  # Deep Anesthesia
    elif bis_value < 60:
        return 1  # Moderate Anesthesia
    elif bis_value < 80:
        return 2  # Light Anesthesia
    else:
        return 3  # Awake

CATEGORY_NAMES = {
    0: "Deep Anesthesia (0-40)",
    1: "Moderate Anesthesia (40-60)",
    2: "Light Anesthesia (60-80)",
    3: "Awake (80-100)"
}


def extract_segments_from_vital(filepath):
    """Extract 20-second ECG/PPG segments with BIS labels from a .vital file."""
    import vitaldb

    filename = os.path.basename(filepath)
    print(f"  Processing {filename}...", end=" ")

    try:
        vf = vitaldb.VitalFile(filepath)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return [], [], [], []

    track_names = vf.get_track_names()

    # Find ECG track
    ecg_track = None
    for name in track_names:
        if 'ECG' in name.upper() and 'WAV' not in name.upper() and 'EEG' not in name.upper():
            ecg_track = name
            break

    # Find PPG/PLETH track (from patient monitor, not X002)
    ppg_track = None
    for name in track_names:
        if 'PLETH' in name.upper() and 'X002' not in name:
            ppg_track = name
            break

    # Find BIS track
    bis_track = None
    for name in track_names:
        if name == 'BIS/BIS':
            bis_track = name
            break
        elif 'BIS' in name.upper() and 'CH' not in name and 'WAV' not in name:
            bis_track = name

    # Find SQI track
    sqi_track = None
    for name in track_names:
        if name == 'BIS/SQI':
            sqi_track = name
            break

    if bis_track is None:
        print("No BIS track found!")
        return [], [], [], []

    if ppg_track is None:
        print("No PPG track found!")
        return [], [], [], []

    # Extract signals at 500Hz
    ppg_data = vf.to_numpy(ppg_track, 1.0 / SAMPLE_RATE).flatten()

    ecg_data = None
    if ecg_track is not None:
        ecg_data = vf.to_numpy(ecg_track, 1.0 / SAMPLE_RATE).flatten()

    # Extract BIS at 1Hz for labeling
    bis_data = vf.to_numpy(bis_track, 1.0).flatten()

    # Extract SQI at 1Hz
    sqi_data = None
    if sqi_track is not None:
        sqi_data = vf.to_numpy(sqi_track, 1.0).flatten()

    total_seconds = len(bis_data)

    ecg_segments = []
    ppg_segments = []
    labels = []
    file_ids = []

    # Sliding window
    start_sec = 0
    while start_sec + WINDOW_SEC <= total_seconds:
        end_sec = start_sec + WINDOW_SEC

        # Get BIS values in this window
        bis_window = bis_data[start_sec:end_sec]
        valid_bis = bis_window[~np.isnan(bis_window)]

        # Check SQI
        if sqi_data is not None:
            sqi_window = sqi_data[start_sec:end_sec]
            valid_sqi = sqi_window[~np.isnan(sqi_window)]
            if len(valid_sqi) > 0 and np.mean(valid_sqi) < MIN_SQI:
                start_sec += STRIDE_SEC
                continue

        # Need at least 10 valid BIS values in 20-second window
        if len(valid_bis) < 10:
            start_sec += STRIDE_SEC
            continue

        # Filter out invalid BIS values (must be 0-100)
        valid_bis = valid_bis[(valid_bis >= 0) & (valid_bis <= 100)]
        if len(valid_bis) < 10:
            start_sec += STRIDE_SEC
            continue

        # BIS label = median of valid values in window
        bis_median = np.median(valid_bis)
        label = bis_to_category(bis_median)

        # Get waveform data
        wav_start = start_sec * SAMPLE_RATE
        wav_end = end_sec * SAMPLE_RATE

        ppg_seg = ppg_data[wav_start:wav_end]

        # Check for NaN in waveforms
        if np.sum(np.isnan(ppg_seg)) > WINDOW_SAMPLES * 0.1:  # allow up to 10% NaN
            start_sec += STRIDE_SEC
            continue

        # Interpolate small NaN gaps
        ppg_seg = _interpolate_nans(ppg_seg)

        ecg_seg = None
        if ecg_data is not None:
            ecg_seg = ecg_data[wav_start:wav_end]
            if len(ecg_seg) < WINDOW_SAMPLES:
                start_sec += STRIDE_SEC
                continue
            if np.sum(np.isnan(ecg_seg)) > WINDOW_SAMPLES * 0.1:
                ecg_seg = None  # ECG not usable, but PPG might be
            else:
                ecg_seg = _interpolate_nans(ecg_seg)

        if len(ppg_seg) < WINDOW_SAMPLES:
            start_sec += STRIDE_SEC
            continue

        ecg_segments.append(ecg_seg if ecg_seg is not None else np.zeros(WINDOW_SAMPLES))
        ppg_segments.append(ppg_seg[:WINDOW_SAMPLES])
        labels.append(label)
        file_ids.append(filename)

        start_sec += STRIDE_SEC

    n_seg = len(labels)
    has_ecg = ecg_track is not None
    print(f"{n_seg} segments (ECG: {'Yes' if has_ecg else 'No'})")

    return ecg_segments, ppg_segments, labels, file_ids


def _interpolate_nans(signal):
    """Linearly interpolate NaN values in a 1D signal."""
    nans = np.isnan(signal)
    if not np.any(nans):
        return signal
    signal = signal.copy()
    x = np.arange(len(signal))
    signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
    return signal


def load_all_data():
    """Load and segment all .vital files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_file = os.path.join(OUTPUT_DIR, "segments_cache.npz")

    if os.path.exists(cache_file):
        print("Loading cached segments...")
        data = np.load(cache_file, allow_pickle=True)
        return data['ecg'], data['ppg'], data['labels'], data['file_ids']

    vital_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.vital")))
    print(f"Found {len(vital_files)} .vital files\n")

    all_ecg, all_ppg, all_labels, all_files = [], [], [], []

    for vf in vital_files:
        ecg, ppg, labels, fids = extract_segments_from_vital(vf)
        all_ecg.extend(ecg)
        all_ppg.extend(ppg)
        all_labels.extend(labels)
        all_files.extend(fids)

    all_ecg = np.array(all_ecg, dtype=np.float32)
    all_ppg = np.array(all_ppg, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_files = np.array(all_files)

    # Save cache
    np.savez_compressed(cache_file, ecg=all_ecg, ppg=all_ppg,
                        labels=all_labels, file_ids=all_files)
    print(f"\nCached to {cache_file}")

    return all_ecg, all_ppg, all_labels, all_files


# ──────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION (for ML baseline)
# ──────────────────────────────────────────────────────────

def extract_features(ecg_segments, ppg_segments):
    """Extract handcrafted features from ECG and PPG for ML models."""
    from scipy import signal as scipy_signal
    from scipy.stats import skew, kurtosis

    features_list = []

    for i in range(len(ppg_segments)):
        feats = {}
        ppg = ppg_segments[i]
        ecg = ecg_segments[i]

        for name, sig in [('ppg', ppg), ('ecg', ecg)]:
            # Time-domain features
            feats[f'{name}_mean'] = np.mean(sig)
            feats[f'{name}_std'] = np.std(sig)
            feats[f'{name}_skew'] = skew(sig)
            feats[f'{name}_kurtosis'] = kurtosis(sig)
            feats[f'{name}_min'] = np.min(sig)
            feats[f'{name}_max'] = np.max(sig)
            feats[f'{name}_ptp'] = np.ptp(sig)
            feats[f'{name}_rms'] = np.sqrt(np.mean(sig**2))
            feats[f'{name}_zcr'] = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)

            # Frequency-domain features
            freqs, psd = scipy_signal.welch(sig, fs=SAMPLE_RATE, nperseg=1024)

            # Band powers
            total_power = np.sum(psd)
            if total_power > 0:
                # VLF (0-0.04 Hz), LF (0.04-0.15 Hz), HF (0.15-0.4 Hz)
                vlf_mask = (freqs >= 0) & (freqs < 0.04)
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                mid_mask = (freqs >= 0.4) & (freqs < 4.0)
                high_mask = (freqs >= 4.0) & (freqs < 40.0)

                feats[f'{name}_vlf_power'] = np.sum(psd[vlf_mask]) / total_power
                feats[f'{name}_lf_power'] = np.sum(psd[lf_mask]) / total_power
                feats[f'{name}_hf_power'] = np.sum(psd[hf_mask]) / total_power
                feats[f'{name}_mid_power'] = np.sum(psd[mid_mask]) / total_power
                feats[f'{name}_high_power'] = np.sum(psd[high_mask]) / total_power

                if np.sum(psd[hf_mask]) > 0:
                    feats[f'{name}_lf_hf_ratio'] = np.sum(psd[lf_mask]) / np.sum(psd[hf_mask])
                else:
                    feats[f'{name}_lf_hf_ratio'] = 0

                feats[f'{name}_spectral_entropy'] = -np.sum(
                    (psd / total_power) * np.log2(psd / total_power + 1e-12)
                )
                feats[f'{name}_dominant_freq'] = freqs[np.argmax(psd)]
            else:
                feats[f'{name}_vlf_power'] = 0
                feats[f'{name}_lf_power'] = 0
                feats[f'{name}_hf_power'] = 0
                feats[f'{name}_mid_power'] = 0
                feats[f'{name}_high_power'] = 0
                feats[f'{name}_lf_hf_ratio'] = 0
                feats[f'{name}_spectral_entropy'] = 0
                feats[f'{name}_dominant_freq'] = 0

        features_list.append(feats)

    return pd.DataFrame(features_list)


# ──────────────────────────────────────────────────────────
# 3. DEEP LEARNING MODEL (1D CNN + Transformer)
# ──────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class DoADataset(Dataset):
    """Dataset for ECG+PPG segments with BIS labels."""

    def __init__(self, ecg, ppg, labels, use_ecg=True):
        self.labels = torch.LongTensor(labels)
        self.use_ecg = use_ecg

        # Normalize per-segment
        ppg_norm = self._normalize(ppg)

        if use_ecg:
            ecg_norm = self._normalize(ecg)
            # Stack as 2-channel input: [N, 2, WINDOW_SAMPLES]
            self.data = torch.FloatTensor(
                np.stack([ecg_norm, ppg_norm], axis=1)
            )
        else:
            # PPG only: [N, 1, WINDOW_SAMPLES]
            self.data = torch.FloatTensor(ppg_norm[:, np.newaxis, :])

    def _normalize(self, x):
        """Z-score normalization per segment."""
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True) + 1e-8
        return (x - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CNNTransformerDoA(nn.Module):
    """
    1D CNN feature extractor + Transformer encoder for DoA prediction.

    Architecture:
    - 1D CNN blocks to downsample and extract local features
    - Positional encoding
    - Transformer encoder for temporal context
    - Classification head
    """

    def __init__(self, in_channels=2, num_classes=4, d_model=128, nhead=8,
                 num_transformer_layers=4, dropout=0.3):
        super().__init__()

        # CNN Feature Extractor (downsamples 10000 -> ~156 timesteps)
        self.cnn = nn.Sequential(
            # Block 1: in_channels -> 32, /4
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            # Block 2: 32 -> 64, /4
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            # Block 3: 64 -> d_model, /4
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=500)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch, channels, time]
        features = self.cnn(x)          # [batch, d_model, seq_len]
        features = features.permute(0, 2, 1)  # [batch, seq_len, d_model]
        features = self.pos_encoding(features)
        features = self.transformer(features)  # [batch, seq_len, d_model]

        # Global average pooling over time
        out = features.mean(dim=1)      # [batch, d_model]
        logits = self.classifier(out)   # [batch, num_classes]
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────
# 4. TRAINING PIPELINE
# ──────────────────────────────────────────────────────────

def train_deep_learning(ecg_segments, ppg_segments, labels, file_ids, use_ecg=True):
    """Train 1D CNN + Transformer model with patient-level split."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    mode_name = "ECG+PPG" if use_ecg else "PPG-only"
    print(f"\n{'='*60}")
    print(f"Training CNN-Transformer Model ({mode_name})")
    print(f"{'='*60}")

    in_channels = 2 if use_ecg else 1

    # Patient-level GroupKFold (3-fold for small dataset)
    unique_files = np.unique(file_ids)
    n_folds = min(5, len(unique_files))
    gkf = GroupKFold(n_splits=n_folds)

    all_preds = np.zeros(len(labels), dtype=np.int64)
    all_probs = np.zeros((len(labels), 4), dtype=np.float32)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(ppg_segments, labels, file_ids)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        print(f"  Train: {len(train_idx)} segments, Val: {len(val_idx)} segments")

        # Count class distribution
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        for c in range(4):
            print(f"    Class {c} ({CATEGORY_NAMES[c]}): "
                  f"train={np.sum(train_labels==c)}, val={np.sum(val_labels==c)}")

        # Create datasets
        train_dataset = DoADataset(ecg_segments[train_idx], ppg_segments[train_idx],
                                   labels[train_idx], use_ecg=use_ecg)
        val_dataset = DoADataset(ecg_segments[val_idx], ppg_segments[val_idx],
                                 labels[val_idx], use_ecg=use_ecg)

        # Class weights for imbalanced data
        class_counts = np.bincount(train_labels, minlength=4).astype(float)
        class_counts = np.maximum(class_counts, 1)  # avoid div by zero
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * 4
        class_weights = torch.FloatTensor(class_weights).to(device)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=0, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

        # Model
        model = CNNTransformerDoA(
            in_channels=in_channels,
            num_classes=4,
            d_model=128,
            nhead=8,
            num_transformer_layers=4,
            dropout=0.3
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_f1 = 0
        patience = 15
        patience_counter = 0
        best_state = None

        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(batch_y)
                train_correct += (logits.argmax(1) == batch_y).sum().item()
                train_total += len(batch_y)

            scheduler.step()

            # Validation
            model.eval()
            val_preds_fold = []
            val_true_fold = []
            val_loss = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * len(batch_y)
                    val_preds_fold.extend(logits.argmax(1).cpu().numpy())
                    val_true_fold.extend(batch_y.cpu().numpy())

            val_f1 = f1_score(val_true_fold, val_preds_fold, average='macro', zero_division=0)
            val_acc = accuracy_score(val_true_fold, val_preds_fold)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: train_loss={train_loss/train_total:.4f}, "
                      f"train_acc={train_correct/train_total:.4f}, "
                      f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model and get predictions
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = F.softmax(logits, dim=1)

        # Collect all val predictions
        idx = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                batch_size = len(batch_y)
                all_preds[val_idx[idx:idx+batch_size]] = preds
                all_probs[val_idx[idx:idx+batch_size]] = probs
                idx += batch_size

        fold_f1 = f1_score(labels[val_idx], all_preds[val_idx], average='macro', zero_division=0)
        fold_acc = accuracy_score(labels[val_idx], all_preds[val_idx])
        fold_metrics.append({'fold': fold+1, 'accuracy': fold_acc, 'f1_macro': fold_f1})
        print(f"  Fold {fold+1} Best: acc={fold_acc:.4f}, f1={fold_f1:.4f}")

    # Overall results
    print(f"\n{'='*60}")
    print(f"CNN-Transformer Results ({mode_name}) - {n_folds}-Fold CV")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy_score(labels, all_preds):.4f}")
    print(f"Overall Macro F1: {f1_score(labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"Overall Weighted F1: {f1_score(labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"\nPer-fold metrics:")
    for m in fold_metrics:
        print(f"  Fold {m['fold']}: acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")
    print(f"Mean Accuracy: {np.mean([m['accuracy'] for m in fold_metrics]):.4f} "
          f"± {np.std([m['accuracy'] for m in fold_metrics]):.4f}")
    print(f"Mean Macro F1: {np.mean([m['f1_macro'] for m in fold_metrics]):.4f} "
          f"± {np.std([m['f1_macro'] for m in fold_metrics]):.4f}")

    print(f"\nClassification Report:")
    target_names = [CATEGORY_NAMES[i] for i in range(4)]
    print(classification_report(labels, all_preds, target_names=target_names, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(labels, all_preds)
    print(cm)

    # Save final model (train on all data)
    print(f"\nTraining final model on all data...")
    full_dataset = DoADataset(ecg_segments, ppg_segments, labels, use_ecg=use_ecg)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=0)

    final_model = CNNTransformerDoA(
        in_channels=in_channels, num_classes=4,
        d_model=128, nhead=8, num_transformer_layers=4, dropout=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Use overall class weights
    class_counts = np.bincount(labels, minlength=4).astype(float)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 4
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    final_model.train()
    for epoch in range(60):
        for batch_x, batch_y in full_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = final_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model_path = os.path.join(OUTPUT_DIR, f"cnn_transformer_{mode_name.replace('+','_').lower()}.pt")
    torch.save(final_model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

    return all_preds, all_probs, fold_metrics


def train_ml_baseline(ecg_segments, ppg_segments, labels, file_ids):
    """Train XGBoost/Random Forest on handcrafted features."""
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*60}")
    print("Extracting Handcrafted Features for ML Baseline...")
    print(f"{'='*60}")

    features_df = extract_features(ecg_segments, ppg_segments)
    print(f"Feature matrix: {features_df.shape}")

    # Replace inf/nan
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = features_df.values

    unique_files = np.unique(file_ids)
    n_folds = min(5, len(unique_files))
    gkf = GroupKFold(n_splits=n_folds)

    # Try multiple models
    models = {
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        ),
    }

    for model_name, model_fn in models.items():
        print(f"\n--- {model_name} ---")
        all_preds = np.zeros(len(labels), dtype=np.int64)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, labels, file_ids)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])

            model = model_fn()
            model.fit(X_train, labels[train_idx])
            all_preds[val_idx] = model.predict(X_val)

        acc = accuracy_score(labels, all_preds)
        f1 = f1_score(labels, all_preds, average='macro', zero_division=0)
        f1_w = f1_score(labels, all_preds, average='weighted', zero_division=0)

        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {f1:.4f}")
        print(f"  Weighted F1: {f1_w:.4f}")

        target_names = [CATEGORY_NAMES[i] for i in range(4)]
        print(classification_report(labels, all_preds, target_names=target_names, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, all_preds))

    # Save best model (RF usually works well)
    print("\nTraining final RandomForest on all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    final_rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    final_rf.fit(X_scaled, labels)

    # Feature importance
    importances = pd.Series(final_rf.feature_importances_, index=features_df.columns)
    print("\nTop 15 Features:")
    print(importances.nlargest(15).to_string())

    with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), 'wb') as f:
        pickle.dump({'model': final_rf, 'scaler': scaler, 'feature_names': list(features_df.columns)}, f)
    print(f"RandomForest model saved to {OUTPUT_DIR}/rf_model.pkl")


# ──────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BIS Prediction from ECG/PPG - Depth of Anesthesia")
    print("=" * 60)
    print(f"Window: {WINDOW_SEC}s, Stride: {STRIDE_SEC}s, Sample Rate: {SAMPLE_RATE}Hz")
    print()

    # Step 1: Load data
    ecg_segments, ppg_segments, labels, file_ids = load_all_data()

    print(f"\nTotal segments: {len(labels)}")
    print("Class distribution:")
    for c in range(4):
        count = np.sum(labels == c)
        pct = count / len(labels) * 100
        print(f"  {CATEGORY_NAMES[c]}: {count} ({pct:.1f}%)")

    if len(labels) < 20:
        print("\nERROR: Too few segments for training. Check data.")
        exit(1)

    # Step 2: ML Baseline (RandomForest + GradientBoosting)
    train_ml_baseline(ecg_segments, ppg_segments, labels, file_ids)

    # Step 3: Deep Learning - ECG + PPG
    print("\n")
    train_deep_learning(ecg_segments, ppg_segments, labels, file_ids, use_ecg=True)

    # Step 4: Deep Learning - PPG only
    print("\n")
    train_deep_learning(ecg_segments, ppg_segments, labels, file_ids, use_ecg=False)

    print("\n" + "=" * 60)
    print("DONE! All models trained and saved.")
    print("=" * 60)
