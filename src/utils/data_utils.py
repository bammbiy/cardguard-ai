"""
데이터 관련 유틸 함수들

여기저기서 반복되는 로드/분할 코드 묶어놓은 것
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_processed_data(data_dir: str):
    """
    처리된 피처 파일 한번에 로드
    generate_data.py 돌리고 나서 쓰는 함수
    """
    path = Path(data_dir)

    agg_df = pd.read_csv(path / 'aggregate_features.csv')
    sequences = np.load(path / 'sequences.npy')
    seq_labels = np.load(path / 'seq_labels.npy')

    return agg_df, sequences, seq_labels


def train_val_split(
    agg_df: pd.DataFrame,
    sequences: np.ndarray,
    seq_labels: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """
    train / val 분리
    시퀀스랑 집계 피처 크기가 다를 수 있어서 각각 따로 처리
    """
    rng = np.random.default_rng(random_state)

    # 집계 피처 분리
    n_agg = len(agg_df)
    val_n = int(n_agg * val_ratio)
    idx = rng.permutation(n_agg)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    train_agg = agg_df.iloc[train_idx].reset_index(drop=True)
    val_agg = agg_df.iloc[val_idx].reset_index(drop=True)

    # 시퀀스 분리
    n_seq = len(sequences)
    val_n_seq = int(n_seq * val_ratio)
    seq_idx = rng.permutation(n_seq)
    val_seq_idx, train_seq_idx = seq_idx[:val_n_seq], seq_idx[val_n_seq:]

    train_seq = sequences[train_seq_idx]
    val_seq = sequences[val_seq_idx]
    train_seq_labels = seq_labels[train_seq_idx]
    val_seq_labels = seq_labels[val_seq_idx]

    return train_agg, val_agg, train_seq, val_seq, train_seq_labels, val_seq_labels


def print_dataset_stats(agg_df: pd.DataFrame, seq_labels: np.ndarray):
    """데이터셋 기본 통계 출력"""
    agg_labels = agg_df['label'].values if 'label' in agg_df.columns else None

    print("=" * 40)
    print("데이터셋 통계")
    print("=" * 40)
    print(f"집계 피처: {agg_df.shape[0]}명")
    if agg_labels is not None:
        normal = (agg_labels == 0).sum()
        abnormal = (agg_labels == 1).sum()
        print(f"  정상:  {normal} ({normal/len(agg_labels)*100:.1f}%)")
        print(f"  이상:  {abnormal} ({abnormal/len(agg_labels)*100:.1f}%)")

    print(f"\n시퀀스: {len(seq_labels)}개")
    if len(seq_labels) > 0:
        normal_s = (seq_labels == 0).sum()
        abnormal_s = (seq_labels == 1).sum()
        print(f"  정상:  {normal_s} ({normal_s/len(seq_labels)*100:.1f}%)")
        print(f"  이상:  {abnormal_s} ({abnormal_s/len(seq_labels)*100:.1f}%)")
    print("=" * 40)
