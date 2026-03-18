"""
.vital 파일 분석 도구 - BIS/마취 심도 예측
============================================
사용법:
    python analyze_vital.py <파일경로.vital>
    python analyze_vital.py <파일경로.vital> --output report.html
    python analyze_vital.py <파일경로.vital> --csv results.csv

출력:
    - 20초 구간별 마취 심도 예측 (Deep/Moderate/Light/Awake)
    - 실제 BIS 값과 예측 비교 (BIS 데이터가 있는 경우)
    - 시계열 시각화 그래프
    - HTML 리포트 또는 CSV 내보내기
"""

import os
import sys
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import timedelta

# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────

SAMPLE_RATE = 500
WINDOW_SEC = 20
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
STRIDE_SEC = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "processed_data")

CATEGORY_NAMES = {
    0: "Deep Anesthesia",
    1: "Moderate Anesthesia",
    2: "Light Anesthesia",
    3: "Awake"
}
CATEGORY_BIS_RANGE = {
    0: "BIS 0-40",
    1: "BIS 40-60",
    2: "BIS 60-80",
    3: "BIS 80-100"
}
CATEGORY_COLORS = {
    0: "#1a237e",   # 진한 남색 - Deep
    1: "#2e7d32",   # 초록 - Moderate
    2: "#f57f17",   # 주황 - Light
    3: "#c62828",   # 빨강 - Awake
}


# ──────────────────────────────────────────────────────────
# 모델 정의 (학습 코드와 동일)
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


# ---- V1 Architecture (matches cnn_transformer_ecg_ppg.pt / ppg-only.pt) ----

class CNNTransformerDoA_V1(nn.Module):
    def __init__(self, in_channels=2, num_classes=4, d_model=128, nhead=8,
                 num_transformer_layers=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=500)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)
        f = self.pos_encoding(f)
        f = self.transformer(f)
        out = f.mean(dim=1)
        return self.classifier(out)


# ---- V2 Architecture (matches cnn_transformer_v2_*.pt) ----

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


class CNNTransformerDoA_V2(nn.Module):
    def __init__(self, in_channels=2, num_classes=4, d_model=128, nhead=8,
                 num_transformer_layers=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, 15, 2, 7),
            nn.BatchNorm1d(32), nn.GELU(),
            ResBlock1D(32, 32, 2), ResBlock1D(32, 64, 2),
            ResBlock1D(64, 64, 2), ResBlock1D(64, d_model, 2),
            ResBlock1D(d_model, d_model, 2), nn.Dropout(dropout * 0.5),
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=500)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)
        f = self.pos_encoding(f)
        f = self.transformer(f)
        avg_pool = f.mean(dim=1)
        max_pool = f.max(dim=1).values
        return self.classifier(torch.cat([avg_pool, max_pool], dim=1))


# ──────────────────────────────────────────────────────────
# .vital 파일 파싱
# ──────────────────────────────────────────────────────────

def parse_vital_file(filepath):
    """Parse a .vital file and extract ECG, PPG, BIS signals."""
    import vitaldb

    print(f"[1/4] 파일 로딩: {os.path.basename(filepath)}")
    vf = vitaldb.VitalFile(filepath)
    track_names = vf.get_track_names()

    print(f"       발견된 트랙: {len(track_names)}개")

    # ECG 트랙 탐색
    ecg_track = None
    for name in track_names:
        if 'ECG' in name.upper() and 'WAV' not in name.upper() and 'EEG' not in name.upper():
            ecg_track = name
            break

    # PPG/PLETH 트랙 탐색
    ppg_track = None
    for name in track_names:
        if 'PLETH' in name.upper() and 'X002' not in name and 'SPO2' not in name.upper():
            ppg_track = name
            break

    # BIS 트랙 탐색 (실제 값이 있으면 비교용)
    bis_track = None
    for name in track_names:
        if name == 'BIS/BIS':
            bis_track = name
            break

    sqi_track = None
    for name in track_names:
        if name == 'BIS/SQI':
            sqi_track = name
            break

    has_ecg = ecg_track is not None
    has_ppg = ppg_track is not None
    has_bis = bis_track is not None

    print(f"       ECG: {'O (' + ecg_track + ')' if has_ecg else 'X'}")
    print(f"       PPG: {'O (' + ppg_track + ')' if has_ppg else 'X'}")
    print(f"       BIS: {'O (비교용)' if has_bis else 'X (예측만 수행)'}")

    if not has_ppg:
        print("\n[ERROR] PPG(PLETH) 신호를 찾을 수 없습니다.")
        print(f"  사용 가능한 트랙: {track_names}")
        sys.exit(1)

    # 신호 추출
    ppg_data = vf.to_numpy(ppg_track, 1.0 / SAMPLE_RATE).flatten()
    ecg_data = vf.to_numpy(ecg_track, 1.0 / SAMPLE_RATE).flatten() if has_ecg else None
    bis_data = vf.to_numpy(bis_track, 1.0).flatten() if has_bis else None
    sqi_data = vf.to_numpy(sqi_track, 1.0).flatten() if sqi_track else None

    total_seconds = len(ppg_data) // SAMPLE_RATE
    print(f"       기록 시간: {total_seconds // 60}분 {total_seconds % 60}초")

    return {
        'ppg': ppg_data,
        'ecg': ecg_data,
        'bis': bis_data,
        'sqi': sqi_data,
        'has_ecg': has_ecg,
        'has_bis': has_bis,
        'total_seconds': total_seconds,
        'filename': os.path.basename(filepath),
    }


def _interpolate_nans(signal):
    nans = np.isnan(signal)
    if not np.any(nans):
        return signal
    if np.all(nans):
        return np.zeros_like(signal)
    signal = signal.copy()
    x = np.arange(len(signal))
    signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
    return signal


def create_segments(data):
    """Create 20-second segments from parsed data."""
    print(f"[2/4] 20초 세그먼트 생성 (stride={STRIDE_SEC}s)...")

    ppg = data['ppg']
    ecg = data['ecg']
    bis = data['bis']
    sqi = data['sqi']
    total_sec = data['total_seconds']

    ecg_segments = []
    ppg_segments = []
    bis_labels = []       # 실제 BIS 중앙값 (있을 때만)
    time_stamps = []      # 각 세그먼트의 시작 시간 (초)
    quality_flags = []    # 신호 품질 플래그

    start_sec = 0
    while start_sec + WINDOW_SEC <= total_sec:
        end_sec = start_sec + WINDOW_SEC
        wav_start = start_sec * SAMPLE_RATE
        wav_end = end_sec * SAMPLE_RATE

        # PPG 추출
        ppg_seg = ppg[wav_start:wav_end]
        if len(ppg_seg) < WINDOW_SAMPLES:
            start_sec += STRIDE_SEC
            continue

        nan_ratio = np.sum(np.isnan(ppg_seg)) / WINDOW_SAMPLES
        if nan_ratio > 0.3:
            start_sec += STRIDE_SEC
            continue

        ppg_seg = _interpolate_nans(ppg_seg[:WINDOW_SAMPLES])

        # ECG 추출
        ecg_seg = np.zeros(WINDOW_SAMPLES)
        if ecg is not None:
            e = ecg[wav_start:wav_end]
            if len(e) >= WINDOW_SAMPLES and np.sum(np.isnan(e[:WINDOW_SAMPLES])) < WINDOW_SAMPLES * 0.3:
                ecg_seg = _interpolate_nans(e[:WINDOW_SAMPLES])

        # BIS 레이블 (비교용)
        bis_val = np.nan
        if bis is not None:
            bw = bis[start_sec:end_sec]
            valid = bw[(~np.isnan(bw)) & (bw >= 0) & (bw <= 100)]
            if len(valid) >= 5:
                bis_val = np.median(valid)

        # SQI 품질
        quality = "good"
        if sqi is not None:
            sq = sqi[start_sec:end_sec]
            vsq = sq[~np.isnan(sq)]
            if len(vsq) > 0 and np.mean(vsq) < 50:
                quality = "low_sqi"
        if nan_ratio > 0.1:
            quality = "noisy"

        ecg_segments.append(ecg_seg)
        ppg_segments.append(ppg_seg)
        bis_labels.append(bis_val)
        time_stamps.append(start_sec)
        quality_flags.append(quality)

        start_sec += STRIDE_SEC

    ecg_segments = np.array(ecg_segments, dtype=np.float32)
    ppg_segments = np.array(ppg_segments, dtype=np.float32)
    bis_labels = np.array(bis_labels, dtype=np.float32)
    time_stamps = np.array(time_stamps)

    print(f"       생성된 세그먼트: {len(time_stamps)}개")
    print(f"       품질 양호: {sum(1 for q in quality_flags if q == 'good')}개")

    return {
        'ecg': ecg_segments,
        'ppg': ppg_segments,
        'bis': bis_labels,
        'times': time_stamps,
        'quality': quality_flags,
        'has_ecg': data['has_ecg'],
        'has_bis': data['has_bis'],
        'filename': data['filename'],
        'total_seconds': data['total_seconds'],
    }


# ──────────────────────────────────────────────────────────
# 모델 로딩 & 추론
# ──────────────────────────────────────────────────────────

def load_model(use_ecg=True):
    """Load trained CNN-Transformer model (auto-detect v1 or v2)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # v2 모델 우선, 없으면 v1
    if use_ecg:
        candidates = [
            ("cnn_transformer_v2_ecg_ppg.pt", "v2"),
            ("cnn_transformer_ecg_ppg.pt", "v1"),
        ]
    else:
        candidates = [
            ("cnn_transformer_v2_ppg-only.pt", "v2"),
            ("cnn_transformer_ppg-only.pt", "v1"),
        ]

    model_path = None
    version = None
    for fname, ver in candidates:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            model_path = p
            version = ver
            break

    if model_path is None:
        return None, device

    in_channels = 2 if use_ecg else 1

    if version == "v2":
        model = CNNTransformerDoA_V2(in_channels=in_channels, num_classes=4,
                                      d_model=128, nhead=8, num_transformer_layers=4, dropout=0.3)
    else:
        model = CNNTransformerDoA_V1(in_channels=in_channels, num_classes=4,
                                      d_model=128, nhead=8, num_transformer_layers=4, dropout=0.3)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"       모델 로딩: {os.path.basename(model_path)} ({version}, {device})")
    return model, device


def predict(segments):
    """Run DoA prediction on segments."""
    print(f"[3/4] 마취 심도 예측 중...")

    ecg_segs = segments['ecg']
    ppg_segs = segments['ppg']
    has_ecg = segments['has_ecg']

    # Z-score normalization
    def normalize(x):
        m = np.mean(x, axis=1, keepdims=True)
        s = np.std(x, axis=1, keepdims=True) + 1e-8
        return (x - m) / s

    ppg_norm = normalize(ppg_segs)

    # Try ECG+PPG model first
    model_ecg_ppg, device = load_model(use_ecg=True)
    model_ppg_only, _ = load_model(use_ecg=False)

    results = {}

    if model_ecg_ppg is not None and has_ecg:
        ecg_norm = normalize(ecg_segs)
        data_tensor = torch.FloatTensor(np.stack([ecg_norm, ppg_norm], axis=1))
        preds, probs = _run_inference(model_ecg_ppg, data_tensor, device)
        results['ecg_ppg'] = {'preds': preds, 'probs': probs, 'name': 'ECG+PPG'}

    if model_ppg_only is not None:
        data_tensor = torch.FloatTensor(ppg_norm[:, np.newaxis, :])
        preds, probs = _run_inference(model_ppg_only, data_tensor, device)
        results['ppg_only'] = {'preds': preds, 'probs': probs, 'name': 'PPG-only'}

    if not results:
        print("       [WARNING] 학습된 모델을 찾을 수 없습니다.")
        print(f"       모델 경로: {MODEL_DIR}")
        sys.exit(1)

    # Primary prediction: ECG+PPG if available, else PPG-only
    primary_key = 'ecg_ppg' if 'ecg_ppg' in results else 'ppg_only'
    primary = results[primary_key]

    segments['predictions'] = primary['preds']
    segments['probabilities'] = primary['probs']
    segments['model_used'] = primary['name']
    segments['all_results'] = results

    # 통계 출력
    preds = primary['preds']
    print(f"       사용 모델: {primary['name']}")
    print(f"       예측 분포:")
    for c in range(4):
        cnt = np.sum(preds == c)
        pct = cnt / len(preds) * 100 if len(preds) > 0 else 0
        print(f"         {CATEGORY_NAMES[c]} ({CATEGORY_BIS_RANGE[c]}): {cnt}개 ({pct:.1f}%)")

    return segments


def _run_inference(model, data_tensor, device, batch_size=128):
    """Run model inference in batches."""
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_probs)


# ──────────────────────────────────────────────────────────
# 시각화 & 리포트
# ──────────────────────────────────────────────────────────

def generate_report(segments, output_path=None):
    """Generate visual report."""
    print(f"[4/4] 리포트 생성 중...")

    times = segments['times']
    preds = segments['predictions']
    probs = segments['probabilities']
    bis = segments['bis']
    quality = segments['quality']
    has_bis = segments['has_bis']
    filename = segments['filename']

    # 시간을 분 단위로 변환
    times_min = times / 60.0
    center_times_min = (times + WINDOW_SEC / 2) / 60.0

    n_plots = 3 if has_bis else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots), sharex=True)
    fig.suptitle(f'마취 심도 분석 - {filename}', fontsize=14, fontweight='bold', y=0.98)

    # ---- Plot 1: 예측된 마취 심도 ----
    ax1 = axes[0]
    colors = [CATEGORY_COLORS[p] for p in preds]
    ax1.scatter(center_times_min, preds, c=colors, s=15, alpha=0.7, zorder=3)

    # 배경 색 구분
    for i in range(len(times) - 1):
        ax1.axvspan(times_min[i], times_min[i] + WINDOW_SEC / 60,
                    alpha=0.15, color=CATEGORY_COLORS[preds[i]], linewidth=0)

    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['Deep\n(0-40)', 'Moderate\n(40-60)', 'Light\n(60-80)', 'Awake\n(80-100)'])
    ax1.set_ylabel('Predicted DoA')
    ax1.set_title('Predicted Depth of Anesthesia')
    ax1.set_ylim(-0.5, 3.5)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # Legend
    patches = [mpatches.Patch(color=CATEGORY_COLORS[i], label=f'{CATEGORY_NAMES[i]} ({CATEGORY_BIS_RANGE[i]})')
               for i in range(4)]
    ax1.legend(handles=patches, loc='upper right', fontsize=8)

    # ---- Plot 2: 예측 확률 ----
    ax2 = axes[1]
    for c in range(4):
        ax2.plot(center_times_min, probs[:, c], color=CATEGORY_COLORS[c],
                 label=CATEGORY_NAMES[c], alpha=0.8, linewidth=1.2)
    ax2.set_ylabel('Probability')
    ax2.set_title('Prediction Probabilities')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Plot 3: 실제 BIS와 비교 (있을 때만) ----
    if has_bis:
        ax3 = axes[2]
        valid_mask = ~np.isnan(bis)

        if np.any(valid_mask):
            ax3.plot(center_times_min[valid_mask], bis[valid_mask],
                     'k-', linewidth=1.5, label='Actual BIS', alpha=0.8)

            # 예측 BIS 구간의 중앙값을 표시
            pred_bis_mid = np.array([20, 50, 70, 90])[preds]
            ax3.scatter(center_times_min, pred_bis_mid, c=colors, s=10,
                        alpha=0.5, label='Predicted BIS range midpoint')

        # BIS 구간 배경
        ax3.axhspan(0, 40, alpha=0.08, color=CATEGORY_COLORS[0])
        ax3.axhspan(40, 60, alpha=0.08, color=CATEGORY_COLORS[1])
        ax3.axhspan(60, 80, alpha=0.08, color=CATEGORY_COLORS[2])
        ax3.axhspan(80, 100, alpha=0.08, color=CATEGORY_COLORS[3])

        ax3.set_ylabel('BIS Value')
        ax3.set_title('Actual BIS vs Predicted')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 정확도 계산 (BIS가 있는 구간)
        if np.any(valid_mask):
            actual_cats = np.array([
                0 if b < 40 else 1 if b < 60 else 2 if b < 80 else 3
                for b in bis[valid_mask]
            ])
            pred_cats = preds[valid_mask]
            accuracy = np.mean(actual_cats == pred_cats)
            ax3.text(0.02, 0.95, f'Accuracy: {accuracy:.1%}',
                     transform=ax3.transAxes, fontsize=11, fontweight='bold',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    axes[-1].set_xlabel('Time (minutes)')

    plt.tight_layout()

    # 저장
    if output_path is None:
        base = os.path.splitext(filename)[0]
        output_path = os.path.join(SCRIPT_DIR, f"{base}_analysis.png")

    if output_path.endswith('.html'):
        # HTML 리포트
        png_path = output_path.replace('.html', '.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        _generate_html_report(segments, png_path, output_path)
    else:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close()
    print(f"       저장 완료: {output_path}")

    return output_path


def _generate_html_report(segments, png_path, html_path):
    """Generate HTML report with embedded image and statistics."""
    import base64

    with open(png_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    times = segments['times']
    preds = segments['predictions']
    probs = segments['probabilities']
    bis = segments['bis']
    filename = segments['filename']
    model_used = segments['model_used']
    total_sec = segments['total_seconds']

    # 통계
    stats = {}
    for c in range(4):
        cnt = np.sum(preds == c)
        duration = cnt * STRIDE_SEC  # approximate
        stats[c] = {'count': int(cnt), 'pct': cnt / len(preds) * 100, 'duration_min': duration / 60}

    # BIS 비교 통계
    bis_stats = ""
    if segments['has_bis']:
        valid = ~np.isnan(bis)
        if np.any(valid):
            actual_cats = np.array([0 if b < 40 else 1 if b < 60 else 2 if b < 80 else 3 for b in bis[valid]])
            accuracy = np.mean(actual_cats == preds[valid]) * 100
            bis_stats = f"""
            <div class="stat-card highlight">
                <h3>실제 BIS 대비 정확도</h3>
                <p class="big-number">{accuracy:.1f}%</p>
                <p>BIS 데이터가 있는 {np.sum(valid)}개 구간 기준</p>
            </div>"""

    rows = ""
    for i in range(len(times)):
        t_start = str(timedelta(seconds=int(times[i])))
        t_end = str(timedelta(seconds=int(times[i] + WINDOW_SEC)))
        cat = preds[i]
        prob = probs[i, cat] * 100
        bis_val = f"{bis[i]:.1f}" if not np.isnan(bis[i]) else "-"
        q = segments['quality'][i]
        q_badge = '<span class="badge good">Good</span>' if q == 'good' else f'<span class="badge warn">{q}</span>'

        rows += f"""<tr>
            <td>{t_start} - {t_end}</td>
            <td><span class="cat-badge" style="background:{CATEGORY_COLORS[cat]}">{CATEGORY_NAMES[cat]}</span></td>
            <td>{prob:.1f}%</td>
            <td>{bis_val}</td>
            <td>{q_badge}</td>
        </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>마취 심도 분석 - {filename}</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #333; margin-top: 30px; }}
    .meta {{ color: #666; margin-bottom: 20px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
    .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #ccc; }}
    .stat-card.deep {{ border-left-color: {CATEGORY_COLORS[0]}; }}
    .stat-card.moderate {{ border-left-color: {CATEGORY_COLORS[1]}; }}
    .stat-card.light {{ border-left-color: {CATEGORY_COLORS[2]}; }}
    .stat-card.awake {{ border-left-color: {CATEGORY_COLORS[3]}; }}
    .stat-card.highlight {{ border-left-color: #ff6f00; background: #fff8e1; }}
    .stat-card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #666; }}
    .big-number {{ font-size: 28px; font-weight: bold; margin: 5px 0; }}
    img {{ width: 100%; border-radius: 5px; margin: 20px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 13px; }}
    th {{ background: #1a237e; color: white; padding: 10px; text-align: left; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
    tr:hover {{ background: #f5f5f5; }}
    .cat-badge {{ color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; }}
    .badge {{ padding: 2px 8px; border-radius: 10px; font-size: 11px; }}
    .badge.good {{ background: #c8e6c9; color: #2e7d32; }}
    .badge.warn {{ background: #fff3e0; color: #e65100; }}
    .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #eee; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
    <h1>마취 심도 (DoA) 분석 리포트</h1>
    <div class="meta">
        <strong>파일:</strong> {filename} |
        <strong>총 기록 시간:</strong> {total_sec // 60}분 {total_sec % 60}초 |
        <strong>분석 구간:</strong> {len(times)}개 (20초 윈도우) |
        <strong>모델:</strong> CNN-Transformer ({model_used})
    </div>

    <h2>요약 통계</h2>
    <div class="stats-grid">
        <div class="stat-card deep">
            <h3>Deep Anesthesia (BIS 0-40)</h3>
            <p class="big-number">{stats[0]['count']}개</p>
            <p>{stats[0]['pct']:.1f}% | ~{stats[0]['duration_min']:.1f}분</p>
        </div>
        <div class="stat-card moderate">
            <h3>Moderate Anesthesia (BIS 40-60)</h3>
            <p class="big-number">{stats[1]['count']}개</p>
            <p>{stats[1]['pct']:.1f}% | ~{stats[1]['duration_min']:.1f}분</p>
        </div>
        <div class="stat-card light">
            <h3>Light Anesthesia (BIS 60-80)</h3>
            <p class="big-number">{stats[2]['count']}개</p>
            <p>{stats[2]['pct']:.1f}% | ~{stats[2]['duration_min']:.1f}분</p>
        </div>
        <div class="stat-card awake">
            <h3>Awake (BIS 80-100)</h3>
            <p class="big-number">{stats[3]['count']}개</p>
            <p>{stats[3]['pct']:.1f}% | ~{stats[3]['duration_min']:.1f}분</p>
        </div>
        {bis_stats}
    </div>

    <h2>시계열 분석</h2>
    <img src="data:image/png;base64,{img_b64}" alt="Analysis Chart">

    <h2>구간별 상세 결과</h2>
    <table>
        <thead>
            <tr><th>시간 구간</th><th>예측 심도</th><th>확신도</th><th>실제 BIS</th><th>신호 품질</th></tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <div class="footer">
        CNN-Transformer 모델 기반 마취 심도 예측 | 20초 윈도우, {SAMPLE_RATE}Hz 샘플링
    </div>
</div>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def export_csv(segments, csv_path):
    """Export results to CSV."""
    import csv
    times = segments['times']
    preds = segments['predictions']
    probs = segments['probabilities']
    bis = segments['bis']
    quality = segments['quality']

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['start_sec', 'end_sec', 'start_time', 'end_time',
                         'predicted_class', 'predicted_label', 'predicted_bis_range',
                         'prob_deep', 'prob_moderate', 'prob_light', 'prob_awake',
                         'confidence', 'actual_bis', 'signal_quality'])

        for i in range(len(times)):
            start = int(times[i])
            end = start + WINDOW_SEC
            cat = preds[i]
            writer.writerow([
                start, end,
                str(timedelta(seconds=start)),
                str(timedelta(seconds=end)),
                cat,
                CATEGORY_NAMES[cat],
                CATEGORY_BIS_RANGE[cat],
                f"{probs[i, 0]:.4f}",
                f"{probs[i, 1]:.4f}",
                f"{probs[i, 2]:.4f}",
                f"{probs[i, 3]:.4f}",
                f"{probs[i, cat]:.4f}",
                f"{bis[i]:.1f}" if not np.isnan(bis[i]) else "",
                quality[i]
            ])

    print(f"       CSV 저장: {csv_path}")


# ──────────────────────────────────────────────────────────
# 콘솔 출력
# ──────────────────────────────────────────────────────────

def print_summary(segments):
    """Print analysis summary to console."""
    times = segments['times']
    preds = segments['predictions']
    probs = segments['probabilities']
    bis = segments['bis']
    filename = segments['filename']
    model_used = segments['model_used']

    print(f"\n{'=' * 60}")
    print(f"  분석 결과: {filename}")
    print(f"  모델: CNN-Transformer ({model_used})")
    print(f"{'=' * 60}")

    # 요약 통계
    print(f"\n  [마취 심도 분포]")
    for c in range(4):
        cnt = np.sum(preds == c)
        pct = cnt / len(preds) * 100 if len(preds) > 0 else 0
        duration = cnt * STRIDE_SEC / 60
        bar = '#' * int(pct / 2)
        print(f"    {CATEGORY_NAMES[c]:25s} ({CATEGORY_BIS_RANGE[c]:10s}): "
              f"{cnt:4d}개 ({pct:5.1f}%) ~{duration:5.1f}분 {bar}")

    # BIS 비교
    if segments['has_bis']:
        valid = ~np.isnan(bis)
        if np.any(valid):
            actual_cats = np.array([0 if b < 40 else 1 if b < 60 else 2 if b < 80 else 3 for b in bis[valid]])
            acc = np.mean(actual_cats == preds[valid]) * 100
            print(f"\n  [실제 BIS 대비 정확도]: {acc:.1f}% ({np.sum(valid)}개 구간)")

    # 시간대별 요약 (10분 단위)
    print(f"\n  [시간대별 요약 (주요 상태)]")
    max_time = int(times[-1]) + WINDOW_SEC if len(times) > 0 else 0
    for t_start in range(0, max_time, 600):  # 10분 간격
        t_end = t_start + 600
        mask = (times >= t_start) & (times < t_end)
        if np.any(mask):
            p = preds[mask]
            dominant = np.bincount(p, minlength=4).argmax()
            avg_prob = probs[mask].mean(axis=0)
            t_str = f"{t_start // 60:3d}-{min(t_end, max_time) // 60:3d}분"
            print(f"    {t_str}: {CATEGORY_NAMES[dominant]:25s} "
                  f"(확신도 {avg_prob[dominant]:.1%})")

    print(f"\n{'=' * 60}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='마취 심도(DoA) 분석 - .vital 파일에서 ECG/PPG 신호로 BIS 구간 예측',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python analyze_vital.py patient01.vital
  python analyze_vital.py patient01.vital --output report.html
  python analyze_vital.py patient01.vital --csv results.csv
  python analyze_vital.py patient01.vital --output report.html --csv results.csv
        """
    )
    parser.add_argument('vital_file', help='.vital 파일 경로')
    parser.add_argument('--output', '-o', help='출력 파일 경로 (.png 또는 .html)')
    parser.add_argument('--csv', help='CSV 결과 내보내기 경로')

    args = parser.parse_args()

    if not os.path.exists(args.vital_file):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {args.vital_file}")
        sys.exit(1)

    if not args.vital_file.lower().endswith('.vital'):
        print(f"[WARNING] .vital 파일이 아닐 수 있습니다: {args.vital_file}")

    # Pipeline
    data = parse_vital_file(args.vital_file)
    segments = create_segments(data)

    if len(segments['times']) == 0:
        print("\n[ERROR] 분석 가능한 세그먼트가 없습니다.")
        sys.exit(1)

    segments = predict(segments)
    print_summary(segments)

    # 시각화
    output_path = args.output
    if output_path is None:
        base = os.path.splitext(os.path.basename(args.vital_file))[0]
        output_path = os.path.join(SCRIPT_DIR, f"{base}_analysis.png")
    generate_report(segments, output_path)

    # CSV
    if args.csv:
        export_csv(segments, args.csv)

    print(f"\n완료!")


if __name__ == "__main__":
    main()
