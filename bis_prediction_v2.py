"""
BIS Prediction from ECG/PPG - Depth of Anesthesia (v2)
======================================================
Improvements over v1:
- Focal Loss for class imbalance
- Oversampling minority classes
- Longer warmup + training
- Label smoothing
- Mixup augmentation
- GradientBoosting with proper tuning
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
import pandas as pd

CATEGORY_NAMES = {
    0: "Deep (0-40)",
    1: "Moderate (40-60)",
    2: "Light (60-80)",
    3: "Awake (80-100)"
}
SAMPLE_RATE = 500
OUTPUT_DIR = "processed_data"


# ──────────────────────────────────────────────────────────
# DATASET with augmentation
# ──────────────────────────────────────────────────────────

class DoADataset(Dataset):
    def __init__(self, ecg, ppg, labels, use_ecg=True, augment=False):
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.use_ecg = use_ecg

        ppg_norm = self._normalize(ppg)
        if use_ecg:
            ecg_norm = self._normalize(ecg)
            self.data = torch.FloatTensor(np.stack([ecg_norm, ppg_norm], axis=1))
        else:
            self.data = torch.FloatTensor(ppg_norm[:, np.newaxis, :])

    def _normalize(self, x):
        m = np.mean(x, axis=1, keepdims=True)
        s = np.std(x, axis=1, keepdims=True) + 1e-8
        return (x - m) / s

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.augment:
            # Random scaling
            if torch.rand(1) < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1)
                x = x * scale
            # Add Gaussian noise
            if torch.rand(1) < 0.5:
                noise = torch.randn_like(x) * 0.05
                x = x + noise
            # Random time shift (circular)
            if torch.rand(1) < 0.3:
                shift = torch.randint(-500, 500, (1,)).item()
                x = torch.roll(x, shift, dims=-1)
        return x, y


# ──────────────────────────────────────────────────────────
# FOCAL LOSS
# ──────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight
        focal_weight = (1 - probs) ** self.gamma

        loss = -focal_weight * smooth_targets * log_probs

        if self.alpha is not None:
            alpha_weight = self.alpha[targets].unsqueeze(1)
            loss = loss * alpha_weight

        return loss.sum(dim=1).mean()


# ──────────────────────────────────────────────────────────
# MODEL: 1D CNN + Transformer
# ──────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride),
            nn.BatchNorm1d(out_ch)
        ) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + self.shortcut(x))


class CNNTransformerDoA(nn.Module):
    def __init__(self, in_channels=2, num_classes=4, d_model=128, nhead=8,
                 num_transformer_layers=4, dropout=0.3):
        super().__init__()

        # ResNet-style CNN backbone (10000 -> ~156 timesteps)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, 15, 2, 7),  # /2
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResBlock1D(32, 32, 2),    # /2 -> 2500
            ResBlock1D(32, 64, 2),    # /2 -> 1250
            ResBlock1D(64, 64, 2),    # /2 -> 625
            ResBlock1D(64, d_model, 2),  # /2 -> ~312
            ResBlock1D(d_model, d_model, 2),  # /2 -> ~156
            nn.Dropout(dropout * 0.5),
        )

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=500)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Dual pooling (avg + max)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)
        f = self.pos_encoding(f)
        f = self.transformer(f)
        # Dual pooling
        avg_pool = f.mean(dim=1)
        max_pool = f.max(dim=1).values
        out = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(out)


# ──────────────────────────────────────────────────────────
# FEATURE EXTRACTION for ML
# ──────────────────────────────────────────────────────────

def extract_features(ecg_segs, ppg_segs):
    features_list = []
    for i in range(len(ppg_segs)):
        feats = {}
        for name, sig in [('ppg', ppg_segs[i]), ('ecg', ecg_segs[i])]:
            # Time domain
            feats[f'{name}_mean'] = np.mean(sig)
            feats[f'{name}_std'] = np.std(sig)
            feats[f'{name}_skew'] = skew(sig)
            feats[f'{name}_kurtosis'] = kurtosis(sig)
            feats[f'{name}_ptp'] = np.ptp(sig)
            feats[f'{name}_rms'] = np.sqrt(np.mean(sig ** 2))
            feats[f'{name}_zcr'] = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)

            # Hjorth parameters
            diff1 = np.diff(sig)
            diff2 = np.diff(diff1)
            var0 = np.var(sig)
            var1 = np.var(diff1)
            var2 = np.var(diff2)
            feats[f'{name}_hjorth_activity'] = var0
            feats[f'{name}_hjorth_mobility'] = np.sqrt(var1 / (var0 + 1e-12))
            feats[f'{name}_hjorth_complexity'] = np.sqrt(var2 / (var1 + 1e-12)) / (np.sqrt(var1 / (var0 + 1e-12)) + 1e-12)

            # Sample entropy approximation (permutation entropy)
            feats[f'{name}_perm_entropy'] = _permutation_entropy(sig, order=3, delay=1)

            # Frequency domain
            freqs, psd = scipy_signal.welch(sig, fs=SAMPLE_RATE, nperseg=1024)
            tp = np.sum(psd)
            if tp > 0:
                for bname, lo, hi in [('vlf', 0, 0.04), ('lf', 0.04, 0.15), ('hf', 0.15, 0.4),
                                       ('mid', 0.4, 4), ('high', 4, 40), ('vhigh', 40, 100)]:
                    mask = (freqs >= lo) & (freqs < hi)
                    feats[f'{name}_{bname}_power'] = np.sum(psd[mask]) / tp
                hf_p = np.sum(psd[(freqs >= 0.15) & (freqs < 0.4)])
                feats[f'{name}_lf_hf_ratio'] = np.sum(psd[(freqs >= 0.04) & (freqs < 0.15)]) / (hf_p + 1e-12)
                feats[f'{name}_spectral_entropy'] = -np.sum((psd / tp) * np.log2(psd / tp + 1e-12))
                feats[f'{name}_dominant_freq'] = freqs[np.argmax(psd)]
                feats[f'{name}_median_freq'] = freqs[np.searchsorted(np.cumsum(psd) / tp, 0.5)]
                feats[f'{name}_spectral_edge_95'] = freqs[np.searchsorted(np.cumsum(psd) / tp, 0.95)]
            else:
                for k in ['vlf_power', 'lf_power', 'hf_power', 'mid_power', 'high_power',
                           'vhigh_power', 'lf_hf_ratio', 'spectral_entropy', 'dominant_freq',
                           'median_freq', 'spectral_edge_95']:
                    feats[f'{name}_{k}'] = 0

        features_list.append(feats)
    return pd.DataFrame(features_list)


def _permutation_entropy(x, order=3, delay=1):
    from itertools import permutations
    n = len(x)
    perms = list(permutations(range(order)))
    c = np.zeros(len(perms))
    for i in range(n - delay * (order - 1)):
        sorted_idx = tuple(np.argsort(x[i:i + delay * order:delay]))
        for j, p in enumerate(perms):
            if sorted_idx == p:
                c[j] += 1
                break
    c = c[c > 0]
    c = c / c.sum()
    return -np.sum(c * np.log2(c))


# ──────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────

def train_dl_model(ecg, ppg, labels, file_ids, use_ecg=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = "ECG+PPG" if use_ecg else "PPG-only"
    in_ch = 2 if use_ecg else 1

    print(f'\n{"=" * 60}')
    print(f'CNN-Transformer ({mode}) - 5-Fold GroupKFold CV [GPU: {device}]')
    print(f'{"=" * 60}')

    n_folds = min(5, len(np.unique(file_ids)))
    gkf = GroupKFold(n_splits=n_folds)
    all_preds = np.zeros(len(labels), dtype=np.int64)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(ppg, labels, file_ids)):
        print(f'\n  Fold {fold + 1}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}')
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        for c in range(4):
            print(f'    Class {c}: train={np.sum(train_labels == c)}, val={np.sum(val_labels == c)}')

        # Weighted random sampler for oversampling minority
        class_counts = np.bincount(train_labels, minlength=4).astype(float)
        class_counts = np.maximum(class_counts, 1)
        sample_weights = 1.0 / class_counts[train_labels]
        sample_weights = sample_weights / sample_weights.sum()
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(train_idx),
            replacement=True
        )

        train_ds = DoADataset(ecg[train_idx], ppg[train_idx], train_labels, use_ecg, augment=True)
        val_ds = DoADataset(ecg[val_idx], ppg[val_idx], val_labels, use_ecg, augment=False)
        train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

        # Focal Loss with class weights
        alpha = torch.FloatTensor(1.0 / class_counts)
        alpha = alpha / alpha.sum() * 4
        criterion = FocalLoss(alpha=alpha.to(device), gamma=2.0, label_smoothing=0.05)

        model = CNNTransformerDoA(in_ch, 4, 128, 8, 4, 0.3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        # Warmup + Cosine schedule
        total_epochs = 80
        warmup_epochs = 5

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_f1 = 0
        patience = 20
        wait = 0
        best_state = None

        for epoch in range(total_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)

                # Mixup augmentation (50% of the time)
                if torch.rand(1).item() < 0.5 and len(by) > 1:
                    lam = np.random.beta(0.4, 0.4)
                    idx_perm = torch.randperm(bx.size(0))
                    bx_mixed = lam * bx + (1 - lam) * bx[idx_perm]
                    optimizer.zero_grad()
                    logits = model(bx_mixed)
                    loss = lam * criterion(logits, by) + (1 - lam) * criterion(logits, by[idx_perm])
                else:
                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = criterion(logits, by)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(by)
                train_correct += (logits.argmax(1) == by).sum().item()
                train_total += len(by)

            scheduler.step()

            # Validation
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    logits = model(bx.to(device))
                    vp.extend(logits.argmax(1).cpu().numpy())
                    vt.extend(by.numpy())

            val_f1 = f1_score(vt, vp, average='macro', zero_division=0)
            val_acc = accuracy_score(vt, vp)

            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch + 1}: loss={train_loss / train_total:.4f}, '
                      f'train_acc={train_correct / train_total:.4f}, '
                      f'val_acc={val_acc:.4f}, val_f1={val_f1:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience and epoch >= 30:
                    print(f'    Early stop epoch {epoch + 1}, best_f1={best_f1:.4f}')
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        idx = 0
        with torch.no_grad():
            for bx, by in val_loader:
                p = model(bx.to(device)).argmax(1).cpu().numpy()
                bs = len(by)
                all_preds[val_idx[idx:idx + bs]] = p
                idx += bs

        fa = accuracy_score(labels[val_idx], all_preds[val_idx])
        ff = f1_score(labels[val_idx], all_preds[val_idx], average='macro', zero_division=0)
        fold_metrics.append((fa, ff))
        print(f'    >> Fold {fold + 1} Best: acc={fa:.4f}, macro_f1={ff:.4f}')

    # Overall results
    print(f'\n{"=" * 60}')
    print(f'FINAL RESULTS - CNN-Transformer ({mode})')
    print(f'{"=" * 60}')
    print(f'Overall Accuracy:   {accuracy_score(labels, all_preds):.4f}')
    print(f'Overall Macro F1:   {f1_score(labels, all_preds, average="macro", zero_division=0):.4f}')
    print(f'Overall Weighted F1:{f1_score(labels, all_preds, average="weighted", zero_division=0):.4f}')
    accs = [m[0] for m in fold_metrics]
    f1s = [m[1] for m in fold_metrics]
    print(f'Mean Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}')
    print(f'Mean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}')
    tn = [CATEGORY_NAMES[i] for i in range(4)]
    print(f'\n{classification_report(labels, all_preds, target_names=tn, zero_division=0)}')
    print('Confusion Matrix:')
    print(confusion_matrix(labels, all_preds))

    # Save final model trained on all data
    print(f'\nTraining final model on all data...')
    full_ds = DoADataset(ecg, ppg, labels, use_ecg, augment=True)
    sw = 1.0 / np.maximum(np.bincount(labels, minlength=4).astype(float), 1)[labels]
    sw = sw / sw.sum()
    full_sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(labels), replacement=True)
    full_loader = DataLoader(full_ds, batch_size=64, sampler=full_sampler, num_workers=0)

    final_model = CNNTransformerDoA(in_ch, 4, 128, 8, 4, 0.3).to(device)
    opt = torch.optim.AdamW(final_model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    cc = np.maximum(np.bincount(labels, minlength=4).astype(float), 1)
    a = torch.FloatTensor(1.0 / cc); a = a / a.sum() * 4
    crit = FocalLoss(alpha=a.to(device), gamma=2.0, label_smoothing=0.05)

    final_model.train()
    for epoch in range(60):
        for bx, by in full_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(final_model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            opt.step()
        sched.step()

    model_path = os.path.join(OUTPUT_DIR, f"cnn_transformer_v2_{mode.replace('+', '_').lower()}.pt")
    torch.save(final_model.state_dict(), model_path)
    print(f'Saved: {model_path}')

    return all_preds, fold_metrics


def train_ml_models(ecg, ppg, labels, file_ids):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    print(f'\n{"=" * 60}')
    print('ML BASELINES (RandomForest + GradientBoosting)')
    print(f'{"=" * 60}')

    print('Extracting features...')
    feat_df = extract_features(ecg, ppg)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = feat_df.values
    print(f'Feature matrix: {X.shape}')

    n_folds = min(5, len(np.unique(file_ids)))
    gkf = GroupKFold(n_splits=n_folds)

    models_cfg = {
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=3,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1
        ),
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42
        ),
    }

    for mname, mfn in models_cfg.items():
        print(f'\n--- {mname} ---')
        all_preds = np.zeros(len(labels), dtype=np.int64)

        for fold, (tr, va) in enumerate(gkf.split(X, labels, file_ids)):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xva = sc.transform(X[va])

            # SMOTE for minority oversampling
            try:
                from imblearn.over_sampling import SMOTE
                min_class_count = min(np.bincount(labels[tr], minlength=4))
                if min_class_count >= 6:
                    sm = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
                    Xtr, ytr = sm.fit_resample(Xtr, labels[tr])
                else:
                    # Use random oversampling instead
                    from imblearn.over_sampling import RandomOverSampler
                    ros = RandomOverSampler(random_state=42)
                    Xtr, ytr = ros.fit_resample(Xtr, labels[tr])
            except ImportError:
                ytr = labels[tr]

            model = mfn()
            model.fit(Xtr, ytr)
            all_preds[va] = model.predict(Xva)

        acc = accuracy_score(labels, all_preds)
        f1m = f1_score(labels, all_preds, average='macro', zero_division=0)
        f1w = f1_score(labels, all_preds, average='weighted', zero_division=0)
        print(f'  Accuracy:    {acc:.4f}')
        print(f'  Macro F1:    {f1m:.4f}')
        print(f'  Weighted F1: {f1w:.4f}')
        tn = [CATEGORY_NAMES[i] for i in range(4)]
        print(classification_report(labels, all_preds, target_names=tn, zero_division=0))
        print('Confusion Matrix:')
        print(confusion_matrix(labels, all_preds))


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BIS Prediction v2 - Depth of Anesthesia")
    print("=" * 60)

    data = np.load(os.path.join(OUTPUT_DIR, 'segments_cache.npz'), allow_pickle=True)
    ecg, ppg, labels, file_ids = data['ecg'], data['ppg'], data['labels'], data['file_ids']

    print(f'Loaded {len(labels)} segments from {len(np.unique(file_ids))} patients')
    for c in range(4):
        cnt = np.sum(labels == c)
        print(f'  {CATEGORY_NAMES[c]}: {cnt} ({cnt / len(labels) * 100:.1f}%)')

    # Install imblearn if needed
    try:
        import imblearn
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'imbalanced-learn'], stdout=subprocess.DEVNULL)

    # 1. ML Baselines
    train_ml_models(ecg, ppg, labels, file_ids)

    # 2. DL - ECG+PPG
    train_dl_model(ecg, ppg, labels, file_ids, use_ecg=True)

    # 3. DL - PPG only
    train_dl_model(ecg, ppg, labels, file_ids, use_ecg=False)

    print(f'\n{"=" * 60}')
    print('ALL DONE!')
    print(f'{"=" * 60}')
