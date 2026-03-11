"""
게임 로그 생성 + 피처 추출까지 한 번에

python scripts/generate_data.py --sessions 5000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.log_generator import generate_sessions
from src.detection.feature_engineering import build_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=2000)
    parser.add_argument('--hands', type=int, default=30)
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--abuser-ratio', type=float, default=0.18)
    parser.add_argument('--raw-output', type=str, default='data/raw')
    parser.add_argument('--processed-output', type=str, default='data/processed')
    args = parser.parse_args()

    print("Step 1: 게임 로그 생성")
    sessions, labels = generate_sessions(
        n_sessions=args.sessions,
        output_dir=args.raw_output,
        n_players=args.players,
        hands_per_session=args.hands,
        abuser_ratio=args.abuser_ratio,
    )

    print("\nStep 2: 피처 엔지니어링")
    agg_df, seq_array, seq_labels = build_dataset(sessions, labels)
    save_dataset(agg_df, seq_array, seq_labels, args.processed_output)

    print("\n완료!")


if __name__ == "__main__":
    main()
