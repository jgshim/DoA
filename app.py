"""
마취 심도(DoA) 분석 웹앱
========================
실행: streamlit run app.py
브라우저에서 .vital 파일을 업로드하면 BIS 구간을 예측합니다.
"""

import os
import io
import tempfile
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F

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
CATEGORY_BIS_RANGE = {0: "BIS 0-40", 1: "BIS 40-60", 2: "BIS 60-80", 3: "BIS 80-100"}
CATEGORY_COLORS = {0: "#1a237e", 1: "#2e7d32", 2: "#f57f17", 3: "#c62828"}
CATEGORY_EMOJIS = {0: "\U0001F535", 1: "\U0001F7E2", 2: "\U0001F7E1", 3: "\U0001F534"}


# ──────────────────────────────────────────────────────────
# 모델 정의
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


class CNNTransformerDoA_V1(nn.Module):
    def __init__(self, in_channels=2, num_classes=4, d_model=128, nhead=8,
                 num_transformer_layers=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, d_model, 3, 2, 1), nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, 2, 1), nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=500)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model * 4, dropout=dropout,
                                         activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        f = self.cnn(x).permute(0, 2, 1)
        f = self.pos_encoding(f)
        f = self.transformer(f)
        return self.classifier(f.mean(dim=1))


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride), nn.BatchNorm1d(out_ch)
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
            nn.Conv1d(in_channels, 32, 15, 2, 7), nn.BatchNorm1d(32), nn.GELU(),
            ResBlock1D(32, 32, 2), ResBlock1D(32, 64, 2),
            ResBlock1D(64, 64, 2), ResBlock1D(64, d_model, 2),
            ResBlock1D(d_model, d_model, 2), nn.Dropout(dropout * 0.5),
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=500)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model * 4, dropout=dropout,
                                         activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_transformer_layers)
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
# 모델 로딩 (캐싱)
# ──────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}

    model_configs = [
        ("cnn_transformer_v2_ecg_ppg.pt", "v2", True, "ECG+PPG (v2)"),
        ("cnn_transformer_ecg_ppg.pt", "v1", True, "ECG+PPG (v1)"),
        ("cnn_transformer_v2_ppg-only.pt", "v2", False, "PPG-only (v2)"),
        ("cnn_transformer_ppg-only.pt", "v1", False, "PPG-only (v1)"),
    ]

    for fname, ver, use_ecg, label in model_configs:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            continue

        key = "ecg_ppg" if use_ecg else "ppg_only"
        if key in models:
            continue  # 이미 더 좋은 버전 로드됨

        in_ch = 2 if use_ecg else 1
        if ver == "v2":
            model = CNNTransformerDoA_V2(in_ch, 4, 128, 8, 4, 0.3)
        else:
            model = CNNTransformerDoA_V1(in_ch, 4, 128, 8, 4, 0.3)

        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models[key] = {"model": model, "label": label, "use_ecg": use_ecg}

    return models, device


# ──────────────────────────────────────────────────────────
# 데이터 처리
# ──────────────────────────────────────────────────────────

def interpolate_nans(signal):
    nans = np.isnan(signal)
    if not np.any(nans):
        return signal
    if np.all(nans):
        return np.zeros_like(signal)
    signal = signal.copy()
    x = np.arange(len(signal))
    signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
    return signal


def parse_vital(uploaded_file):
    import vitaldb

    with tempfile.NamedTemporaryFile(suffix='.vital', delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        vf = vitaldb.VitalFile(tmp_path)
    finally:
        os.unlink(tmp_path)

    tracks = vf.get_track_names()

    ecg_track = ppg_track = bis_track = sqi_track = None
    for t in tracks:
        tu = t.upper()
        if 'ECG' in tu and 'WAV' not in tu and 'EEG' not in tu and ecg_track is None:
            ecg_track = t
        if 'PLETH' in tu and 'X002' not in t and 'SPO2' not in tu and ppg_track is None:
            ppg_track = t
        if t == 'BIS/BIS':
            bis_track = t
        if t == 'BIS/SQI':
            sqi_track = t

    if ppg_track is None:
        return None, "PPG(PLETH) 신호를 찾을 수 없습니다."

    ppg = vf.to_numpy(ppg_track, 1.0 / SAMPLE_RATE).flatten()
    ecg = vf.to_numpy(ecg_track, 1.0 / SAMPLE_RATE).flatten() if ecg_track else None
    bis = vf.to_numpy(bis_track, 1.0).flatten() if bis_track else None
    sqi = vf.to_numpy(sqi_track, 1.0).flatten() if sqi_track else None

    total_sec = len(ppg) // SAMPLE_RATE

    return {
        'ppg': ppg, 'ecg': ecg, 'bis': bis, 'sqi': sqi,
        'has_ecg': ecg_track is not None, 'has_bis': bis_track is not None,
        'total_seconds': total_sec,
        'ecg_track': ecg_track, 'ppg_track': ppg_track,
        'tracks': tracks,
    }, None


def create_segments(data):
    ppg, ecg, bis, sqi = data['ppg'], data['ecg'], data['bis'], data['sqi']
    total_sec = data['total_seconds']

    ecg_segs, ppg_segs, bis_vals, times, qualities = [], [], [], [], []

    s = 0
    while s + WINDOW_SEC <= total_sec:
        ws, we = s * SAMPLE_RATE, (s + WINDOW_SEC) * SAMPLE_RATE
        ppg_seg = ppg[ws:we]
        if len(ppg_seg) < WINDOW_SAMPLES:
            s += STRIDE_SEC; continue

        nan_r = np.sum(np.isnan(ppg_seg)) / WINDOW_SAMPLES
        if nan_r > 0.3:
            s += STRIDE_SEC; continue

        ppg_seg = interpolate_nans(ppg_seg[:WINDOW_SAMPLES])

        ecg_seg = np.zeros(WINDOW_SAMPLES)
        if ecg is not None:
            e = ecg[ws:we]
            if len(e) >= WINDOW_SAMPLES and np.sum(np.isnan(e[:WINDOW_SAMPLES])) < WINDOW_SAMPLES * 0.3:
                ecg_seg = interpolate_nans(e[:WINDOW_SAMPLES])

        bis_val = np.nan
        if bis is not None:
            bw = bis[s:s + WINDOW_SEC]
            valid = bw[(~np.isnan(bw)) & (bw >= 0) & (bw <= 100)]
            if len(valid) >= 5:
                bis_val = np.median(valid)

        quality = "good"
        if sqi is not None:
            sq = sqi[s:s + WINDOW_SEC]
            vsq = sq[~np.isnan(sq)]
            if len(vsq) > 0 and np.mean(vsq) < 50:
                quality = "low_sqi"
        if nan_r > 0.1:
            quality = "noisy"

        ecg_segs.append(ecg_seg)
        ppg_segs.append(ppg_seg)
        bis_vals.append(bis_val)
        times.append(s)
        qualities.append(quality)
        s += STRIDE_SEC

    return {
        'ecg': np.array(ecg_segs, dtype=np.float32),
        'ppg': np.array(ppg_segs, dtype=np.float32),
        'bis': np.array(bis_vals, dtype=np.float32),
        'times': np.array(times),
        'quality': qualities,
        'has_ecg': data['has_ecg'],
        'has_bis': data['has_bis'],
    }


def predict_segments(segments, models, device):
    ecg, ppg = segments['ecg'], segments['ppg']

    def normalize(x):
        m = np.mean(x, axis=1, keepdims=True)
        s = np.std(x, axis=1, keepdims=True) + 1e-8
        return (x - m) / s

    ppg_n = normalize(ppg)
    results = {}

    if 'ecg_ppg' in models and segments['has_ecg']:
        ecg_n = normalize(ecg)
        tensor = torch.FloatTensor(np.stack([ecg_n, ppg_n], axis=1))
        preds, probs = run_inference(models['ecg_ppg']['model'], tensor, device)
        results['ecg_ppg'] = {'preds': preds, 'probs': probs, 'label': models['ecg_ppg']['label']}

    if 'ppg_only' in models:
        tensor = torch.FloatTensor(ppg_n[:, np.newaxis, :])
        preds, probs = run_inference(models['ppg_only']['model'], tensor, device)
        results['ppg_only'] = {'preds': preds, 'probs': probs, 'label': models['ppg_only']['label']}

    primary_key = 'ecg_ppg' if 'ecg_ppg' in results else 'ppg_only'
    return results, primary_key


def run_inference(model, tensor, device, batch_size=128):
    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i + batch_size].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs)
    return np.array(all_preds), np.array(all_probs)


# ──────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────

def plot_timeline(times, preds, probs, bis, has_bis):
    center = (times + WINDOW_SEC / 2) / 60.0
    colors = [CATEGORY_COLORS[p] for p in preds]

    n_plots = 3 if has_bis else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.5 * n_plots), sharex=True)

    # 1) 예측 심도
    ax = axes[0]
    for i in range(len(times) - 1):
        ax.axvspan(times[i] / 60, (times[i] + WINDOW_SEC) / 60,
                   alpha=0.2, color=CATEGORY_COLORS[preds[i]], linewidth=0)
    ax.scatter(center, preds, c=colors, s=12, alpha=0.8, zorder=3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Deep\n(0-40)', 'Moderate\n(40-60)', 'Light\n(60-80)', 'Awake\n(80-100)'])
    ax.set_ylabel('Predicted DoA')
    ax.set_title('Predicted Depth of Anesthesia', fontweight='bold')
    ax.set_ylim(-0.5, 3.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    patches = [mpatches.Patch(color=CATEGORY_COLORS[i], label=f'{CATEGORY_NAMES[i]}')
               for i in range(4)]
    ax.legend(handles=patches, loc='upper right', fontsize=8)

    # 2) 확률
    ax = axes[1]
    for c in range(4):
        ax.plot(center, probs[:, c], color=CATEGORY_COLORS[c],
                label=CATEGORY_NAMES[c], alpha=0.8, linewidth=1.2)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Confidence', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) BIS 비교
    if has_bis:
        ax = axes[2]
        valid = ~np.isnan(bis)
        if np.any(valid):
            ax.plot(center[valid], bis[valid], 'k-', lw=1.5, label='Actual BIS', alpha=0.8)
            pred_mid = np.array([20, 50, 70, 90])[preds]
            ax.scatter(center, pred_mid, c=colors, s=8, alpha=0.4, label='Predicted range mid')
        ax.axhspan(0, 40, alpha=0.06, color=CATEGORY_COLORS[0])
        ax.axhspan(40, 60, alpha=0.06, color=CATEGORY_COLORS[1])
        ax.axhspan(60, 80, alpha=0.06, color=CATEGORY_COLORS[2])
        ax.axhspan(80, 100, alpha=0.06, color=CATEGORY_COLORS[3])
        ax.set_ylabel('BIS Value')
        ax.set_title('Actual BIS vs Predicted', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        if np.any(valid):
            actual_cats = np.array([0 if b < 40 else 1 if b < 60 else 2 if b < 80 else 3
                                    for b in bis[valid]])
            acc = np.mean(actual_cats == preds[valid])
            ax.text(0.02, 0.95, f'Accuracy: {acc:.1%}', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    axes[-1].set_xlabel('Time (minutes)')
    plt.tight_layout()
    return fig


def build_detail_table(times, preds, probs, bis, quality):
    rows = []
    for i in range(len(times)):
        cat = preds[i]
        rows.append({
            'Start': str(timedelta(seconds=int(times[i]))),
            'End': str(timedelta(seconds=int(times[i] + WINDOW_SEC))),
            'Prediction': f"{CATEGORY_EMOJIS[cat]} {CATEGORY_NAMES[cat]}",
            'BIS Range': CATEGORY_BIS_RANGE[cat],
            'Confidence': f"{probs[i, cat]:.1%}",
            'Actual BIS': f"{bis[i]:.1f}" if not np.isnan(bis[i]) else "-",
            'Quality': quality[i],
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="DoA Analyzer",
        page_icon="\U0001FA7A",
        layout="wide",
    )

    st.title("\U0001FA7A Depth of Anesthesia Analyzer")
    st.markdown("**.vital** 파일을 업로드하면 ECG/PPG 신호로부터 마취 심도(BIS 구간)를 예측합니다.")

    # 사이드바
    with st.sidebar:
        st.header("Settings")
        st.markdown(f"**Model Directory:** `{MODEL_DIR}`")

        models, device = load_models()
        st.success(f"Device: **{device}**")

        if models:
            st.markdown("**Loaded Models:**")
            for k, v in models.items():
                st.markdown(f"- {v['label']}")
        else:
            st.error("No models found! Train models first.")
            return

        st.divider()
        st.markdown("""
        **BIS Categories:**
        - \U0001F535 **Deep** (BIS 0-40)
        - \U0001F7E2 **Moderate** (BIS 40-60)
        - \U0001F7E1 **Light** (BIS 60-80)
        - \U0001F534 **Awake** (BIS 80-100)
        """)

    # 파일 업로드
    uploaded = st.file_uploader(
        "**.vital 파일을 업로드하세요**",
        type=['vital'],
        help="VitalDB 형식의 .vital 파일을 지원합니다."
    )

    if uploaded is None:
        st.info("\U0001F4C1 .vital 파일을 드래그하거나 클릭하여 업로드하세요.")
        return

    # 분석 시작
    with st.spinner("파일 분석 중..."):
        # 1. 파싱
        with st.status("\U0001F50D 파일 분석 중...", expanded=True) as status:
            st.write("1\uFE0F\u20E3 파일 로딩...")
            data, error = parse_vital(uploaded)
            if error:
                st.error(f"오류: {error}")
                return

            total_min = data['total_seconds'] // 60
            total_sec = data['total_seconds'] % 60

            col1, col2, col3 = st.columns(3)
            col1.metric("기록 시간", f"{total_min}분 {total_sec}초")
            col2.metric("ECG", "O" if data['has_ecg'] else "X")
            col3.metric("BIS (비교용)", "O" if data['has_bis'] else "X")

            # 2. 세그먼트
            st.write("2\uFE0F\u20E3 20초 세그먼트 생성...")
            segments = create_segments(data)
            n_seg = len(segments['times'])

            if n_seg == 0:
                st.error("분석 가능한 세그먼트가 없습니다.")
                return

            st.write(f"   생성: **{n_seg}개** 세그먼트")

            # 3. 예측
            st.write("3\uFE0F\u20E3 마취 심도 예측...")
            results, primary_key = predict_segments(segments, models, device)

            if not results:
                st.error("모델 예측 실패")
                return

            primary = results[primary_key]
            preds = primary['preds']
            probs = primary['probs']

            st.write(f"   모델: **{primary['label']}**")
            status.update(label="\u2705 분석 완료!", state="complete")

    # ──────────────────────────────────────────
    # 결과 표시
    # ──────────────────────────────────────────

    st.divider()
    st.header("\U0001F4CA 분석 결과")

    # 요약 카드
    cols = st.columns(4)
    for c in range(4):
        cnt = int(np.sum(preds == c))
        pct = cnt / len(preds) * 100
        duration = cnt * STRIDE_SEC / 60
        cols[c].metric(
            label=f"{CATEGORY_EMOJIS[c]} {CATEGORY_NAMES[c]}",
            value=f"{cnt}개 ({pct:.1f}%)",
            delta=f"~{duration:.1f}분",
        )

    # BIS 정확도
    if segments['has_bis']:
        valid = ~np.isnan(segments['bis'])
        if np.any(valid):
            actual_cats = np.array([0 if b < 40 else 1 if b < 60 else 2 if b < 80 else 3
                                    for b in segments['bis'][valid]])
            acc = np.mean(actual_cats == preds[valid]) * 100
            st.metric("\U0001F3AF 실제 BIS 대비 정확도", f"{acc:.1f}%",
                      delta=f"{np.sum(valid)}개 구간 기준")

    # 시계열 그래프
    st.subheader("\U0001F4C8 시계열 분석")
    fig = plot_timeline(segments['times'], preds, probs, segments['bis'], segments['has_bis'])
    st.pyplot(fig)
    plt.close(fig)

    # 시간대별 요약
    st.subheader("\U0001F552 시간대별 요약 (10분 단위)")
    times = segments['times']
    max_time = int(times[-1]) + WINDOW_SEC

    timeline_data = []
    for t_start in range(0, max_time, 600):
        t_end = min(t_start + 600, max_time)
        mask = (times >= t_start) & (times < t_end)
        if np.any(mask):
            p = preds[mask]
            dominant = int(np.bincount(p, minlength=4).argmax())
            avg_conf = float(probs[mask].mean(axis=0)[dominant])
            timeline_data.append({
                'Time': f"{t_start // 60}-{t_end // 60}분",
                'Dominant': f"{CATEGORY_EMOJIS[dominant]} {CATEGORY_NAMES[dominant]}",
                'Confidence': f"{avg_conf:.1%}",
                'Deep': int(np.sum(p == 0)),
                'Moderate': int(np.sum(p == 1)),
                'Light': int(np.sum(p == 2)),
                'Awake': int(np.sum(p == 3)),
            })

    st.dataframe(pd.DataFrame(timeline_data), use_container_width=True, hide_index=True)

    # 상세 테이블
    with st.expander("\U0001F4CB 구간별 상세 결과", expanded=False):
        detail_df = build_detail_table(times, preds, probs, segments['bis'], segments['quality'])
        st.dataframe(detail_df, use_container_width=True, hide_index=True, height=400)

    # CSV 다운로드
    csv_df = pd.DataFrame({
        'start_sec': times.astype(int),
        'end_sec': (times + WINDOW_SEC).astype(int),
        'start_time': [str(timedelta(seconds=int(t))) for t in times],
        'end_time': [str(timedelta(seconds=int(t + WINDOW_SEC))) for t in times],
        'predicted_class': preds,
        'predicted_label': [CATEGORY_NAMES[p] for p in preds],
        'bis_range': [CATEGORY_BIS_RANGE[p] for p in preds],
        'prob_deep': probs[:, 0],
        'prob_moderate': probs[:, 1],
        'prob_light': probs[:, 2],
        'prob_awake': probs[:, 3],
        'confidence': [probs[i, preds[i]] for i in range(len(preds))],
        'actual_bis': segments['bis'],
        'signal_quality': segments['quality'],
    })

    st.download_button(
        label="\U0001F4E5 CSV 다운로드",
        data=csv_df.to_csv(index=False, encoding='utf-8-sig'),
        file_name=f"{uploaded.name.replace('.vital', '')}_doa_results.csv",
        mime='text/csv',
    )

    # 비교 모델 (있으면)
    if len(results) > 1:
        with st.expander("\U0001F504 다른 모델 결과 비교"):
            for key, res in results.items():
                if key == primary_key:
                    continue
                st.markdown(f"**{res['label']}**")
                comp_cols = st.columns(4)
                for c in range(4):
                    cnt = int(np.sum(res['preds'] == c))
                    pct = cnt / len(res['preds']) * 100
                    comp_cols[c].metric(f"{CATEGORY_EMOJIS[c]} {CATEGORY_NAMES[c]}", f"{cnt} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
