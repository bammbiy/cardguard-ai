"""
게임 로그 생성기

지정한 수만큼 게임을 시뮬레이션하고 JSON으로 저장
어뷰저 비율은 현실적으로 15~20% 수준으로 설정

실행:
    python -m src.simulator.log_generator --sessions 5000 --output data/raw/
"""

import json
import uuid
import random
import argparse
from pathlib import Path
from tqdm import tqdm

from .poker_game import PokerGame, GameLog
from .player import NormalPlayer, AbuserPlayer, BotPlayer


def create_table(n_players: int = 6, abuser_ratio: float = 0.18, bot_ratio: float = 0.05):
    """
    테이블 하나 구성
    기본 6명 테이블 기준 (실제 카지노 카드게임 기준)
    """
    players = []
    player_ids = [f"p_{uuid.uuid4().hex[:6]}" for _ in range(n_players)]

    # 어뷰저, 봇 수 결정
    n_abusers = max(0, int(n_players * abuser_ratio))
    n_bots = max(0, int(n_players * bot_ratio))
    n_normal = n_players - n_abusers - n_bots

    # 콜루전 파트너 설정 (어뷰저가 2명 이상이면 첫 2명을 파트너로)
    collude_partner = None
    if n_abusers >= 2:
        collude_partner = player_ids[n_normal + 1]

    for i in range(n_players):
        pid = player_ids[i]

        if i < n_normal:
            # 일반 플레이어 - 성향 랜덤하게 다양하게
            players.append(NormalPlayer(
                player_id=pid,
                aggression=random.uniform(0.2, 0.9),
                bluff_rate=random.uniform(0.05, 0.25)
            ))

        elif i < n_normal + n_abusers:
            # 어뷰저
            abuse_types = ['chip_dump', 'collusion', 'overcall']
            abuse_type = random.choice(abuse_types)
            partner = collude_partner if abuse_type == 'collusion' and i == n_normal else None
            players.append(AbuserPlayer(
                player_id=pid,
                collude_partner=partner,
                abuse_type=abuse_type
            ))

        else:
            # 봇
            players.append(BotPlayer(
                player_id=pid,
                skill_level=random.uniform(0.7, 0.95)
            ))

    return players


def serialize_log(log: GameLog) -> dict:
    """GameLog → JSON 직렬화"""
    return {
        'game_id': log.game_id,
        'players': log.players,
        'actions': log.actions,
        'pot_final': log.pot_final,
        'winner': log.winner,
        'community_cards': log.community_cards,
    }


def get_player_labels(players) -> dict:
    """플레이어 ID → 타입 레이블 맵"""
    return {p.player_id: p.player_type for p in players}


def generate_sessions(
    n_sessions: int,
    output_dir: str,
    n_players: int = 6,
    abuser_ratio: float = 0.18,
    bot_ratio: float = 0.05,
    hands_per_session: int = 30,
):
    """
    n_sessions 개의 세션을 생성해서 output_dir에 저장

    세션 = 같은 테이블에서 연속으로 여러 핸드 플레이
    (실제 게임에서 보통 30~100핸드 정도 하고 테이블 바꿈)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_sessions = []
    labels = {}   # player_id → 'normal'/'abuser'/'bot'

    print(f"세션 {n_sessions}개 생성 중... (테이블당 {hands_per_session}핸드)")

    for _ in tqdm(range(n_sessions), desc="시뮬레이션"):
        session_id = uuid.uuid4().hex[:10]
        players = create_table(n_players, abuser_ratio, bot_ratio)

        # 이번 테이블 레이블 기록
        session_labels = get_player_labels(players)
        labels.update(session_labels)

        game = PokerGame(
            player_ids=[p.player_id for p in players],
            starting_chips=10000,
            big_blind=100
        )

        session_logs = []
        for hand_num in range(hands_per_session):
            game_id = f"{session_id}_h{hand_num:03d}"
            dealer_pos = hand_num % n_players

            try:
                log = game.play_hand(game_id, players, dealer_pos=dealer_pos)
                session_logs.append(serialize_log(log))
            except Exception as e:
                # 간혹 엣지케이스로 에러 날 수 있음 - 그냥 스킵
                continue

        all_sessions.append({
            'session_id': session_id,
            'players': {p.player_id: p.player_type for p in players},
            'hands': session_logs
        })

    # 저장
    logs_file = output_path / 'game_logs.json'
    labels_file = output_path / 'player_labels.json'

    with open(logs_file, 'w', encoding='utf-8') as f:
        json.dump(all_sessions, f, ensure_ascii=False, indent=2)

    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # 간단한 통계 출력
    total_players = len(labels)
    abuser_count = sum(1 for t in labels.values() if t == 'abuser')
    bot_count = sum(1 for t in labels.values() if t == 'bot')
    normal_count = total_players - abuser_count - bot_count

    print(f"\n✅ 생성 완료")
    print(f"  - 총 세션: {len(all_sessions)}개")
    print(f"  - 총 플레이어: {total_players}명")
    print(f"    · 정상: {normal_count} ({normal_count/total_players*100:.1f}%)")
    print(f"    · 어뷰저: {abuser_count} ({abuser_count/total_players*100:.1f}%)")
    print(f"    · 봇: {bot_count} ({bot_count/total_players*100:.1f}%)")
    print(f"  - 저장 경로: {output_path}")

    return all_sessions, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="포커 게임 로그 생성기")
    parser.add_argument('--sessions', type=int, default=1000, help='생성할 세션 수')
    parser.add_argument('--output', type=str, default='data/raw/', help='출력 디렉토리')
    parser.add_argument('--players', type=int, default=6, help='테이블당 플레이어 수')
    parser.add_argument('--hands', type=int, default=30, help='세션당 핸드 수')
    parser.add_argument('--abuser-ratio', type=float, default=0.18, help='어뷰저 비율')
    args = parser.parse_args()

    generate_sessions(
        n_sessions=args.sessions,
        output_dir=args.output,
        n_players=args.players,
        hands_per_session=args.hands,
        abuser_ratio=args.abuser_ratio,
    )
