# Depth of Anesthesia (DoA) Prediction

ECG/PPG 신호로부터 BIS(Bispectral Index) 구간을 예측하는 딥러닝 모델

## BIS Categories
| Category | BIS Range | Description |
|----------|-----------|-------------|
| Deep Anesthesia | 0-40 | 깊은 마취 |
| Moderate Anesthesia | 40-60 | 적정 마취 |
| Light Anesthesia | 60-80 | 얕은 마취 |
| Awake | 80-100 | 각성 |

## Structure
```
├── app.py                  # Streamlit 웹앱 (파일 업로드 → 분석)
├── analyze_vital.py        # CLI 분석 도구
├── bis_prediction.py       # 학습 파이프라인 v1
├── bis_prediction_v2.py    # 학습 파이프라인 v2 (Focal Loss, Oversampling, Mixup)
├── processed_data/         # 학습된 모델 (.pt, .pkl)
└── DoA data/               # .vital 원본 데이터 (git 제외)
```

## Usage

### Web App (Streamlit)
```bash
pip install streamlit vitaldb torch scikit-learn matplotlib
streamlit run app.py
```
브라우저에서 `.vital` 파일을 업로드하면 마취 심도를 분석합니다.

### CLI
```bash
python analyze_vital.py patient.vital
python analyze_vital.py patient.vital --output report.html --csv results.csv
```

## Models
- **CNN-Transformer (ECG+PPG)**: 1D CNN + Transformer Encoder, 20초 윈도우
- **CNN-Transformer (PPG-only)**: PPG만으로 예측
- **RandomForest / GradientBoosting**: Handcrafted feature 기반 ML 베이스라인

## Training
```bash
# 데이터 전처리 + 모델 학습
python bis_prediction_v2.py
```
