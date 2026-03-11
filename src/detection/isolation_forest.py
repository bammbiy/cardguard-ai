"""
Isolation Forest 기반 이상 탐지

비지도 방식이라서 레이블 없이도 돌아가는 게 장점
특히 훈련 데이터에 없는 새로운 패턴의 어뷰저를 잡을 때 유용

오염율(contamination)은 실제 어뷰저 비율에 맞게 설정하는 게 맞는데
실제 서비스에선 모르니까 conservative하게 낮게 잡는 편이 나음
(FP가 FN보다 비용이 작으면 높게 - 비즈니스 결정사항)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score


# IF에 쓸 피처 목록 (집계 피처 중 가장 discriminative한 것들만)
FEATURE_COLS = [
    'fold_rate', 'call_rate', 'raise_rate', 'allin_rate',
    'hs_mean', 'hs_std',
    'strong_fold_rate', 'strong_raise_rate',
    'weak_fold_rate', 'weak_call_rate',
    'avg_call_pot_odds', 'hs_po_corr',
    'raise_ratio_mean', 'raise_ratio_std',
    'avg_allin_hs',
]


class AbuseDetectorIF:
    """
    Isolation Forest 기반 어뷰저 탐지기

    fit() → 정상 플레이어 데이터로 학습
    predict_score() → anomaly score 반환 (-1에 가까울수록 이상)
    predict() → 0 (정상) / 1 (이상) 반환
    """

    def __init__(self, contamination: float = 0.15, n_estimators: int = 200, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples='auto',
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0).values
        return X

    def fit(self, df: pd.DataFrame):
        """
        학습 - 정상 유저 데이터만 넣는 게 이상적
        실제 서비스에선 레이블이 없으니까 전체 데이터로 학습하고
        contamination으로 오염율 지정하는 방식 사용
        """
        X = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        print(f"IF 학습 완료 (n={len(df)}, contamination={self.contamination})")
        return self

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Anomaly score 반환
        IF의 decision_function: 낮을수록 이상 (음수 방향)
        0~1로 변환해서 반환 (1 = 이상)
        """
        if not self.is_fitted:
            raise RuntimeError("먼저 fit()을 호출하세요")
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        # [-0.5, 0.5] 범위를 [0, 1]로 대략 정규화
        normalized = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
        return normalized

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """0 (정상) / 1 (이상) 예측"""
        if not self.is_fitted:
            raise RuntimeError("먼저 fit()을 호출하세요")
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        # sklearn IF: 1=정상, -1=이상 → 0/1로 변환
        return (preds == -1).astype(int)

    def evaluate(self, df: pd.DataFrame, true_labels: np.ndarray = None):
        """성능 평가 - 레이블 있을 때만"""
        scores = self.predict_score(df)
        preds = self.predict(df)

        print("\n=== Isolation Forest 평가 ===")
        if true_labels is not None:
            print(classification_report(true_labels, preds, target_names=['정상', '이상']))
            try:
                auc = roc_auc_score(true_labels, scores)
                print(f"ROC-AUC: {auc:.4f}")
            except Exception:
                pass

        n_flagged = preds.sum()
        print(f"플래그된 유저: {n_flagged}/{len(preds)} ({n_flagged/len(preds)*100:.1f}%)")
        return scores, preds

    def get_feature_importance(self, df: pd.DataFrame, top_n: int = 10):
        """
        피처별 이상 기여도 분석
        각 피처를 하나씩 셔플했을 때 anomaly score 변화로 측정
        (Permutation Importance 방식 응용)
        """
        if not self.is_fitted:
            raise RuntimeError("먼저 fit()을 호출하세요")

        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        base_score = np.mean(np.abs(self.model.decision_function(X_scaled)))

        importances = {}
        cols = [c for c in self.feature_cols if c in df.columns]

        for i, col in enumerate(cols):
            X_perm = X_scaled.copy()
            np.random.shuffle(X_perm[:, i])
            perm_score = np.mean(np.abs(self.model.decision_function(X_perm)))
            importances[col] = abs(perm_score - base_score)

        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        print(f"\n상위 {top_n} 중요 피처:")
        for feat, imp in sorted_imp[:top_n]:
            print(f"  {feat:<30} {imp:.4f}")

        return dict(sorted_imp)

    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'feature_cols': self.feature_cols}, path)
        print(f"모델 저장 → {path}")

    @classmethod
    def load(cls, path: str) -> 'AbuseDetectorIF':
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.feature_cols = data['feature_cols']
        obj.is_fitted = True
        return obj
