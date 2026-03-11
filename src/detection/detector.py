"""
앙상블 탐지기

IF + LSTM 두 모델의 스코어를 합쳐서 최종 판정
단순 OR 방식보다 weighted score가 FP를 줄이는 데 효과적이었음

실제 적용 시 고려사항:
- IF는 빠름 (실시간 OK)
- LSTM은 약간 느림 (배치 처리 권장)
- 긴급 플래그는 IF 단독으로, 최종 판정은 앙상블로
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from typing import Optional, Tuple

from .isolation_forest import AbuseDetectorIF
from .lstm_model import LSTMDetector


class EnsembleDetector:
    """
    IF + LSTM 앙상블 탐지기

    scoring 방식:
    final_score = w_if * if_score + w_lstm * lstm_score
    final_pred = 1 if final_score > threshold else 0
    """

    def __init__(
        self,
        if_weight: float = 0.4,
        lstm_weight: float = 0.6,
        threshold: float = 0.55,
    ):
        self.if_weight = if_weight
        self.lstm_weight = lstm_weight
        self.threshold = threshold

        self.if_detector = AbuseDetectorIF()
        self.lstm_detector = LSTMDetector()

    def fit(
        self,
        agg_df: pd.DataFrame,
        sequences: np.ndarray,
        labels: np.ndarray,
        lstm_epochs: int = 30,
    ):
        """
        두 모델 각각 학습
        LSTM은 정상 시퀀스만 사용
        """
        print("=" * 50)
        print("앙상블 탐지기 학습 시작")
        print("=" * 50)

        # IF 학습 (전체 데이터)
        print("\n[1/2] Isolation Forest 학습...")
        self.if_detector.fit(agg_df)

        # LSTM 학습 (정상 데이터만)
        print("\n[2/2] LSTM Autoencoder 학습...")
        if sequences.size > 0:
            normal_mask = labels == 0
            normal_seqs = sequences[normal_mask]
            print(f"  정상 시퀀스: {len(normal_seqs)}개 사용")
            self.lstm_detector.fit(normal_seqs, epochs=lstm_epochs)

        print("\n앙상블 학습 완료!")
        return self

    def predict_scores(
        self,
        agg_df: pd.DataFrame,
        sequences: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        IF score, LSTM score, 앙상블 score 반환
        주의: agg_df랑 sequences가 플레이어 수가 달라도 됨
              플레이어 단위 집계에서 시퀀스 수가 더 많을 수 있음
        """
        if_scores = self.if_detector.predict_score(agg_df)

        if sequences.size > 0:
            lstm_raw = self.lstm_detector.predict_score(sequences)
            # LSTM 스코어는 시퀀스 단위라서 플레이어 단위로 집계
            # 지금은 agg_df 행 수에 맞게 평균 풀링 (간소화)
            if len(lstm_raw) != len(if_scores):
                # 시퀀스 수가 다를 때 - agg_df 크기에 맞게 리샘플
                step = len(lstm_raw) / len(if_scores)
                indices = [int(i * step) for i in range(len(if_scores))]
                lstm_scores = lstm_raw[indices]
            else:
                lstm_scores = lstm_raw

            # 0~1 정규화
            lstm_scores = (lstm_scores - lstm_scores.min()) / (lstm_scores.max() - lstm_scores.min() + 1e-9)
        else:
            lstm_scores = np.zeros(len(if_scores))

        ensemble_scores = self.if_weight * if_scores + self.lstm_weight * lstm_scores
        return if_scores, lstm_scores, ensemble_scores

    def predict(
        self,
        agg_df: pd.DataFrame,
        sequences: np.ndarray,
    ) -> np.ndarray:
        """최종 예측 (0: 정상, 1: 이상)"""
        _, _, ensemble_scores = self.predict_scores(agg_df, sequences)
        return (ensemble_scores > self.threshold).astype(int)

    def evaluate(
        self,
        agg_df: pd.DataFrame,
        sequences: np.ndarray,
        true_labels: np.ndarray,
    ):
        """세 모델 비교 평가"""
        if_scores, lstm_scores, ensemble_scores = self.predict_scores(agg_df, sequences)

        if_preds = self.if_detector.predict(agg_df)
        lstm_preds = (lstm_scores > 0.5).astype(int) if sequences.size > 0 else np.zeros(len(true_labels))
        ensemble_preds = (ensemble_scores > self.threshold).astype(int)

        print("\n" + "=" * 55)
        print("성능 비교")
        print("=" * 55)

        for name, preds, scores in [
            ("Isolation Forest", if_preds, if_scores),
            ("LSTM Autoencoder", lstm_preds, lstm_scores),
            ("앙상블 (최종)", ensemble_preds, ensemble_scores),
        ]:
            print(f"\n■ {name}")
            print(classification_report(true_labels, preds, target_names=['정상', '이상'], digits=3))
            try:
                auc = roc_auc_score(true_labels, scores)
                print(f"  ROC-AUC: {auc:.4f}")
            except Exception:
                pass

        # 혼동행렬 (앙상블)
        cm = confusion_matrix(true_labels, ensemble_preds)
        print("\n앙상블 혼동행렬:")
        print(f"  TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")
        print(f"  FP Rate (정상을 이상으로): {cm[0,1]/(cm[0,0]+cm[0,1]):.3f}")
        print(f"  FN Rate (이상을 놓침):    {cm[1,0]/(cm[1,0]+cm[1,1]):.3f}")

        return ensemble_scores, ensemble_preds

    def flag_report(self, agg_df: pd.DataFrame, sequences: np.ndarray, top_n: int = 20):
        """
        상위 의심 플레이어 리포트
        실제 운영에서 어드민이 보게 될 형태
        """
        if_scores, lstm_scores, ensemble_scores = self.predict_scores(agg_df, sequences)

        report_df = agg_df[['player_id']].copy() if 'player_id' in agg_df.columns else pd.DataFrame()
        report_df['if_score'] = if_scores
        report_df['lstm_score'] = lstm_scores
        report_df['ensemble_score'] = ensemble_scores
        report_df['flagged'] = (ensemble_scores > self.threshold).astype(int)

        # 위험도 등급
        def risk_level(score):
            if score > 0.8:
                return '🔴 HIGH'
            elif score > 0.65:
                return '🟠 MEDIUM'
            elif score > self.threshold:
                return '🟡 LOW'
            return '🟢 NORMAL'

        report_df['risk'] = report_df['ensemble_score'].apply(risk_level)
        report_df = report_df.sort_values('ensemble_score', ascending=False)

        print(f"\n의심 플레이어 TOP {top_n}")
        print("-" * 70)
        print(report_df.head(top_n).to_string(index=False))
        return report_df

    def save(self, model_dir: str):
        """두 모델 저장"""
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.if_detector.save(str(out / 'isolation_forest.pkl'))
        self.lstm_detector.save(str(out / 'lstm_autoencoder.pt'))
        print(f"앙상블 모델 저장 → {model_dir}")

    @classmethod
    def load(cls, model_dir: str, **kwargs) -> 'EnsembleDetector':
        obj = cls(**kwargs)
        dir_path = Path(model_dir)
        obj.if_detector = AbuseDetectorIF.load(str(dir_path / 'isolation_forest.pkl'))
        obj.lstm_detector = LSTMDetector.load(str(dir_path / 'lstm_autoencoder.pt'))
        return obj
