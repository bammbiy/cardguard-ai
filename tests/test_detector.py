"""
탐지 모델 테스트

IF / LSTM / 앙상블 각각 간단하게 돌려보는 테스트
실제 학습까지 하면 너무 오래 걸리니까 작은 데이터로 smoke test 위주

완벽한 성능 검증보단 "일단 돌아가는지" 확인하는 게 목적
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.detection.isolation_forest import AbuseDetectorIF, FEATURE_COLS
from src.detection.lstm_model import LSTMDetector
from src.detection.detector import EnsembleDetector


def _make_dummy_agg_df(n: int = 100, abnormal_ratio: float = 0.2) -> pd.DataFrame:
    """
    테스트용 가짜 집계 피처 DataFrame
    정상은 낮은 allin_rate / 높은 weak_fold_rate
    이상은 반대 - 탐지가 쉽게 차이를 과장해서 만듦
    """
    np.random.seed(42)
    n_abnormal = int(n * abnormal_ratio)
    n_normal = n - n_abnormal

    def make_rows(n_rows, is_abnormal):
        rows = []
        for _ in range(n_rows):
            base = {f: 0.0 for f in FEATURE_COLS}
            if is_abnormal:
                base.update({
                    'fold_rate': np.random.uniform(0.0, 0.15),
                    'call_rate': np.random.uniform(0.6, 0.95),
                    'raise_rate': np.random.uniform(0.0, 0.1),
                    'check_rate': np.random.uniform(0.0, 0.1),
                    'allin_rate': np.random.uniform(0.2, 0.5),
                    'hs_mean': np.random.uniform(0.3, 0.5),
                    'hs_std': np.random.uniform(0.05, 0.1),
                    'strong_fold_rate': np.random.uniform(0.3, 0.7),  # 강한 핸드에서 폴드
                    'strong_raise_rate': np.random.uniform(0.0, 0.1),
                    'weak_fold_rate': np.random.uniform(0.0, 0.1),    # 약한 핸드에서 폴드 안 함
                    'weak_call_rate': np.random.uniform(0.7, 1.0),
                    'avg_call_pot_odds': np.random.uniform(0.5, 0.8),
                    'hs_po_corr': np.random.uniform(-0.5, 0.0),
                    'raise_ratio_mean': np.random.uniform(1.5, 3.0),
                    'raise_ratio_std': np.random.uniform(0.01, 0.05),  # 봇: 일정한 레이즈
                    'avg_allin_hs': np.random.uniform(0.2, 0.4),
                    'allin_count': np.random.randint(5, 20),
                })
            else:
                base.update({
                    'fold_rate': np.random.uniform(0.3, 0.55),
                    'call_rate': np.random.uniform(0.2, 0.4),
                    'raise_rate': np.random.uniform(0.1, 0.25),
                    'check_rate': np.random.uniform(0.1, 0.3),
                    'allin_rate': np.random.uniform(0.0, 0.05),
                    'hs_mean': np.random.uniform(0.45, 0.65),
                    'hs_std': np.random.uniform(0.15, 0.3),
                    'strong_fold_rate': np.random.uniform(0.0, 0.1),
                    'strong_raise_rate': np.random.uniform(0.5, 0.85),
                    'weak_fold_rate': np.random.uniform(0.6, 0.9),
                    'weak_call_rate': np.random.uniform(0.05, 0.2),
                    'avg_call_pot_odds': np.random.uniform(0.2, 0.4),
                    'hs_po_corr': np.random.uniform(0.2, 0.7),
                    'raise_ratio_mean': np.random.uniform(0.4, 0.8),
                    'raise_ratio_std': np.random.uniform(0.1, 0.3),
                    'avg_allin_hs': np.random.uniform(0.7, 0.95),
                    'allin_count': np.random.randint(0, 3),
                })
            base['label'] = 1 if is_abnormal else 0
            base['player_id'] = f'p_{np.random.randint(10000, 99999)}'
            rows.append(base)
        return rows

    rows = make_rows(n_normal, False) + make_rows(n_abnormal, True)
    return pd.DataFrame(rows)


def _make_dummy_sequences(n: int = 200, seq_len: int = 20, input_dim: int = 12) -> tuple:
    """테스트용 가짜 시퀀스"""
    np.random.seed(42)
    n_abnormal = n // 5
    n_normal = n - n_abnormal

    normal_seqs = np.random.uniform(0, 0.5, (n_normal, seq_len, input_dim))
    abnormal_seqs = np.random.uniform(0.5, 1.0, (n_abnormal, seq_len, input_dim))

    sequences = np.vstack([normal_seqs, abnormal_seqs])
    labels = np.array([0] * n_normal + [1] * n_abnormal)

    return sequences, labels


# ─── Isolation Forest ───────────────────────────────

class TestIsolationForest:
    def test_fit_and_predict_runs(self):
        df = _make_dummy_agg_df(200)
        detector = AbuseDetectorIF(contamination=0.2)
        detector.fit(df)
        preds = detector.predict(df)
        assert len(preds) == len(df)

    def test_predict_returns_binary(self):
        df = _make_dummy_agg_df(100)
        detector = AbuseDetectorIF()
        detector.fit(df)
        preds = detector.predict(df)
        assert set(preds).issubset({0, 1})

    def test_score_range(self):
        df = _make_dummy_agg_df(100)
        detector = AbuseDetectorIF()
        detector.fit(df)
        scores = detector.predict_score(df)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0 + 1e-6

    def test_not_fitted_raises(self):
        df = _make_dummy_agg_df(50)
        detector = AbuseDetectorIF()
        with pytest.raises(RuntimeError):
            detector.predict(df)

    def test_save_and_load(self, tmp_path):
        df = _make_dummy_agg_df(100)
        detector = AbuseDetectorIF()
        detector.fit(df)

        save_path = str(tmp_path / 'if_model.pkl')
        detector.save(save_path)

        loaded = AbuseDetectorIF.load(save_path)
        preds_original = detector.predict(df)
        preds_loaded = loaded.predict(df)
        assert np.array_equal(preds_original, preds_loaded)


# ─── LSTM ────────────────────────────────────────────

class TestLSTMDetector:
    def test_fit_runs(self):
        seqs, labels = _make_dummy_sequences(100)
        normal_seqs = seqs[labels == 0]
        detector = LSTMDetector(input_dim=12, hidden_dim=16, latent_dim=8, seq_len=20)
        # epochs 적게 해서 빠르게
        detector.fit(normal_seqs, epochs=3, batch_size=32, verbose=False)
        assert detector.is_fitted
        assert detector.threshold is not None

    def test_predict_shape(self):
        seqs, labels = _make_dummy_sequences(100)
        normal_seqs = seqs[labels == 0]
        detector = LSTMDetector(input_dim=12, hidden_dim=16, latent_dim=8, seq_len=20)
        detector.fit(normal_seqs, epochs=2, verbose=False)
        preds = detector.predict(seqs)
        assert len(preds) == len(seqs)

    def test_score_is_non_negative(self):
        seqs, labels = _make_dummy_sequences(80)
        normal_seqs = seqs[labels == 0]
        detector = LSTMDetector(input_dim=12, hidden_dim=16, latent_dim=8, seq_len=20)
        detector.fit(normal_seqs, epochs=2, verbose=False)
        scores = detector.predict_score(seqs)
        assert np.all(scores >= 0)

    def test_not_fitted_raises(self):
        seqs, _ = _make_dummy_sequences(20)
        detector = LSTMDetector()
        with pytest.raises(RuntimeError):
            detector.predict(seqs)


# ─── 앙상블 ──────────────────────────────────────────

class TestEnsembleDetector:
    def test_fit_and_evaluate(self):
        df = _make_dummy_agg_df(200)
        seqs, seq_labels = _make_dummy_sequences(200)
        agg_labels = df['label'].values

        detector = EnsembleDetector(if_weight=0.4, lstm_weight=0.6, threshold=0.5)
        detector.fit(df, seqs, seq_labels, lstm_epochs=3)

        preds = detector.predict(df, seqs)
        assert len(preds) == len(df)
        assert set(preds).issubset({0, 1})

    def test_ensemble_score_range(self):
        df = _make_dummy_agg_df(100)
        seqs, seq_labels = _make_dummy_sequences(100)

        detector = EnsembleDetector()
        detector.fit(df, seqs, seq_labels, lstm_epochs=2)

        _, _, ens_scores = detector.predict_scores(df, seqs)
        assert np.all(ens_scores >= 0)

    def test_flag_report_returns_df(self):
        df = _make_dummy_agg_df(100)
        seqs, seq_labels = _make_dummy_sequences(100)

        detector = EnsembleDetector()
        detector.fit(df, seqs, seq_labels, lstm_epochs=2)

        report = detector.flag_report(df, seqs, top_n=10)
        assert isinstance(report, pd.DataFrame)
        assert 'ensemble_score' in report.columns
        assert 'risk' in report.columns

    def test_weight_sum_affects_score(self):
        """가중치 바꾸면 앙상블 스코어도 달라져야"""
        df = _make_dummy_agg_df(100)
        seqs, seq_labels = _make_dummy_sequences(100)

        d1 = EnsembleDetector(if_weight=0.9, lstm_weight=0.1)
        d2 = EnsembleDetector(if_weight=0.1, lstm_weight=0.9)

        d1.fit(df, seqs, seq_labels, lstm_epochs=2)
        d2.fit(df, seqs, seq_labels, lstm_epochs=2)

        _, _, s1 = d1.predict_scores(df, seqs)
        _, _, s2 = d2.predict_scores(df, seqs)

        # 가중치가 다르면 스코어도 달라야
        assert not np.allclose(s1, s2, atol=1e-3)
