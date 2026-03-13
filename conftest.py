"""
pytest 설정 파일

공통 fixture 여기다 모아놓음
테스트 짜다 보니까 같은 더미 데이터 만드는 코드가 계속 반복돼서
fixture로 뺀 거 - 나중에 추가될 테스트도 여기서 가져다 쓰면 됨
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pytest

from src.detection.isolation_forest import FEATURE_COLS


@pytest.fixture(scope='session')
def dummy_agg_df():
    """
    세션 스코프라서 테스트 전체에서 한 번만 생성
    크기 좀 넉넉하게 잡음 - 작으면 IF가 불안정해짐
    """
    np.random.seed(0)
    n = 300
    n_abnormal = 60

    rows = []
    for i in range(n):
        is_ab = i < n_abnormal
        row = {f: 0.0 for f in FEATURE_COLS}
        row.update({
            'fold_rate':        np.random.uniform(0.0, 0.2) if is_ab else np.random.uniform(0.3, 0.6),
            'call_rate':        np.random.uniform(0.6, 0.95) if is_ab else np.random.uniform(0.2, 0.45),
            'raise_rate':       np.random.uniform(0.0, 0.1) if is_ab else np.random.uniform(0.1, 0.3),
            'check_rate':       np.random.uniform(0.0, 0.1) if is_ab else np.random.uniform(0.1, 0.25),
            'allin_rate':       np.random.uniform(0.15, 0.4) if is_ab else np.random.uniform(0.0, 0.05),
            'hs_mean':          np.random.uniform(0.3, 0.5) if is_ab else np.random.uniform(0.45, 0.7),
            'hs_std':           np.random.uniform(0.05, 0.1) if is_ab else np.random.uniform(0.15, 0.3),
            'strong_fold_rate': np.random.uniform(0.3, 0.7) if is_ab else np.random.uniform(0.0, 0.1),
            'strong_raise_rate':np.random.uniform(0.0, 0.1) if is_ab else np.random.uniform(0.5, 0.9),
            'weak_fold_rate':   np.random.uniform(0.0, 0.15) if is_ab else np.random.uniform(0.6, 0.9),
            'weak_call_rate':   np.random.uniform(0.7, 1.0) if is_ab else np.random.uniform(0.05, 0.2),
            'avg_call_pot_odds':np.random.uniform(0.5, 0.8) if is_ab else np.random.uniform(0.2, 0.4),
            'hs_po_corr':       np.random.uniform(-0.5, 0.0) if is_ab else np.random.uniform(0.2, 0.7),
            'raise_ratio_mean': np.random.uniform(1.5, 3.0) if is_ab else np.random.uniform(0.4, 0.8),
            'raise_ratio_std':  np.random.uniform(0.01, 0.04) if is_ab else np.random.uniform(0.1, 0.3),
            'avg_allin_hs':     np.random.uniform(0.2, 0.4) if is_ab else np.random.uniform(0.7, 0.95),
            'allin_count':      float(np.random.randint(5, 20)) if is_ab else float(np.random.randint(0, 3)),
        })
        row['label'] = 1 if is_ab else 0
        row['player_id'] = f'p_{i:04d}'
        rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture(scope='session')
def dummy_sequences():
    """
    세션 스코프 시퀀스 fixture
    (n, seq_len, input_dim) 형태
    """
    np.random.seed(1)
    n, seq_len, input_dim = 400, 20, 12
    n_abnormal = 80

    normal = np.random.uniform(0, 0.45, (n - n_abnormal, seq_len, input_dim))
    abnormal = np.random.uniform(0.55, 1.0, (n_abnormal, seq_len, input_dim))

    seqs = np.vstack([normal, abnormal])
    labels = np.array([0] * (n - n_abnormal) + [1] * n_abnormal)

    return seqs, labels


@pytest.fixture(scope='session')
def fitted_if_detector(dummy_agg_df):
    """미리 학습된 IF 탐지기"""
    from src.detection.isolation_forest import AbuseDetectorIF
    detector = AbuseDetectorIF(contamination=0.2)
    detector.fit(dummy_agg_df)
    return detector


@pytest.fixture(scope='session')
def fitted_lstm_detector(dummy_sequences):
    """미리 학습된 LSTM 탐지기 (epochs 적게)"""
    from src.detection.lstm_model import LSTMDetector
    seqs, labels = dummy_sequences
    normal_seqs = seqs[labels == 0]
    detector = LSTMDetector(input_dim=12, hidden_dim=16, latent_dim=8, seq_len=20)
    detector.fit(normal_seqs, epochs=3, batch_size=64, verbose=False)
    return detector


@pytest.fixture(scope='session')
def fitted_ensemble(dummy_agg_df, dummy_sequences):
    """미리 학습된 앙상블 탐지기"""
    from src.detection.detector import EnsembleDetector
    seqs, seq_labels = dummy_sequences
    detector = EnsembleDetector(if_weight=0.4, lstm_weight=0.6, threshold=0.5)
    detector.fit(dummy_agg_df, seqs, seq_labels, lstm_epochs=3)
    return detector
