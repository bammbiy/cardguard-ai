"""
탐지 실행 스크립트

학습된 모델로 게임 세션 하나를 분석해서 의심 플레이어 리포트 출력
실제 서비스에서 어드민 페이지에 붙이는 용도라고 생각하면 됨

python scripts/run_detector.py --input data/raw/game_logs.json --model models/
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.feature_engineering import (
    extract_player_actions,
    compute_aggregate_features,
    build_sequences,
)
from src.detection.detector import EnsembleDetector

import pandas as pd
import numpy as np


def load_session(input_path: str) -> list:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 단일 세션(dict)이면 리스트로 감싸기
    if isinstance(data, dict):
        return [data]
    return data


def build_features_from_sessions(sessions: list):
    """
    세션 리스트에서 탐지에 필요한 피처 뽑기
    train할 때랑 동일한 파이프라인 써야 함 - 여기서 따로 구현하면 나중에 골치 아픔
    """
    player_actions = extract_player_actions(sessions)

    rows = []
    seq_list = []
    pids = []

    for pid, actions in player_actions.items():
        if len(actions) < 5:
            # 액션 너무 적으면 피처 신뢰도 낮아서 스킵
            continue

        feats = compute_aggregate_features(actions)
        if not feats:
            continue

        feats['player_id'] = pid
        rows.append(feats)

        seqs = build_sequences(actions)
        if seqs.size > 0:
            seq_list.append(seqs)
            pids.append(pid)

    agg_df = pd.DataFrame(rows)
    seq_array = np.vstack(seq_list) if seq_list else np.array([])

    return agg_df, seq_array


def print_summary(report_df: pd.DataFrame, sessions: list):
    """결과 요약 출력 - 터미널에서 보기 좋게"""
    print("\n" + "=" * 60)
    print("CardGuard AI — 탐지 결과 요약")
    print("=" * 60)

    total_hands = sum(len(s.get('hands', [])) for s in sessions)
    flagged = report_df[report_df['flagged'] == 1]

    print(f"  분석 세션:   {len(sessions)}개")
    print(f"  분석 핸드:   {total_hands}개")
    print(f"  분석 플레이어: {len(report_df)}명")
    print(f"  플래그 플레이어: {len(flagged)}명 ({len(flagged)/max(len(report_df),1)*100:.1f}%)")

    if len(flagged) == 0:
        print("\n  이상 행동 플레이어 없음 🟢")
        return

    print(f"\n{'플레이어':<15} {'위험도':<12} {'앙상블 스코어':<14} {'IF 스코어':<12} {'LSTM 스코어'}")
    print("-" * 65)

    for _, row in report_df.head(20).iterrows():
        pid = str(row.get('player_id', 'unknown'))[:14]
        risk = row.get('risk', '-')
        e_score = row.get('ensemble_score', 0)
        i_score = row.get('if_score', 0)
        l_score = row.get('lstm_score', 0)
        print(f"{pid:<15} {risk:<12} {e_score:<14.4f} {i_score:<12.4f} {l_score:.4f}")

    print("\n* 상위 20명만 표시 / 전체 결과는 --output 옵션으로 저장")


def main():
    parser = argparse.ArgumentParser(description="CardGuard AI 탐지기")
    parser.add_argument('--input', type=str, required=True, help='게임 로그 JSON 파일 경로')
    parser.add_argument('--model', type=str, default='models/', help='모델 디렉토리')
    parser.add_argument('--output', type=str, default=None, help='결과 CSV 저장 경로 (선택)')
    parser.add_argument('--threshold', type=float, default=None, help='탐지 임계값 (기본: 학습 시 설정값)')
    args = parser.parse_args()

    # 입력 파일 확인
    if not Path(args.input).exists():
        print(f"오류: 파일을 찾을 수 없습니다 → {args.input}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"오류: 모델 디렉토리를 찾을 수 없습니다 → {args.model}")
        print("먼저 scripts/train_model.py 를 실행하세요")
        sys.exit(1)

    # 로드
    print(f"로그 로드 중... ({args.input})")
    sessions = load_session(args.input)
    print(f"  {len(sessions)}개 세션 로드 완료")

    # 피처 추출
    print("피처 추출 중...")
    agg_df, seq_array = build_features_from_sessions(sessions)
    print(f"  플레이어 {len(agg_df)}명 분석 대상")

    if len(agg_df) == 0:
        print("분석 가능한 플레이어가 없습니다. (액션 수 부족)")
        sys.exit(0)

    # 모델 로드
    print(f"모델 로드 중... ({args.model})")
    kwargs = {}
    if args.threshold is not None:
        kwargs['threshold'] = args.threshold
    detector = EnsembleDetector.load(args.model, **kwargs)

    # 탐지 실행
    print("탐지 실행 중...")
    report_df = detector.flag_report(agg_df, seq_array, top_n=len(agg_df))

    # 결과 출력
    print_summary(report_df, sessions)

    # CSV 저장 (옵션)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_path, index=False)
        print(f"\n결과 저장 완료 → {args.output}")


if __name__ == "__main__":
    main()
