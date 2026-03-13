"""
모델 학습 스크립트

python scripts/train_model.py --data data/processed --output models/
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import EnsembleDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed')
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--if-weight', type=float, default=0.4)
    parser.add_argument('--lstm-weight', type=float, default=0.6)
    parser.add_argument('--threshold', type=float, default=0.55)
    args = parser.parse_args()

    data_path = Path(args.data)

    print("데이터 로드 중...")
    agg_df = pd.read_csv(data_path / 'aggregate_features.csv')
    sequences = np.load(data_path / 'sequences.npy')
    seq_labels = np.load(data_path / 'seq_labels.npy')
    agg_labels = agg_df['label'].values

    print(f"집계 피처: {agg_df.shape}")
    print(f"시퀀스: {sequences.shape}")
    print(f"이상 비율 (집계): {agg_labels.mean():.2%}")
    print(f"이상 비율 (시퀀스): {seq_labels.mean():.2%}")

    # 학습
    detector = EnsembleDetector(
        if_weight=args.if_weight,
        lstm_weight=args.lstm_weight,
        threshold=args.threshold,
    )

    detector.fit(agg_df, sequences, seq_labels, lstm_epochs=args.epochs)

    # 평가
    print("\n학습 데이터 기준 성능 (참고용 - 실제론 validation set 분리 필요):")
    detector.evaluate(agg_df, sequences, agg_labels)

    # 의심 플레이어 리포트
    detector.flag_report(agg_df, sequences, top_n=15)

    # 저장
    detector.save(args.output)
    print(f"\n모델 저장 완료 → {args.output}")


if __name__ == "__main__":
    main()
