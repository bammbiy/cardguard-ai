"""
텍사스 홀덤 게임 엔진

룰 기준은 기본 캐시 게임 기준으로 짰습니다.
토너먼트 룰 (앤티, 레벨업 등) 은 지금은 없음 - 나중에 추가할 수도
"""

import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# 카드 관련 상수
SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}


class Street(Enum):
    PREFLOP = "preflop"
    FLOP    = "flop"
    TURN    = "turn"
    RIVER   = "river"


class Action(Enum):
    FOLD  = "fold"
    CHECK = "check"
    CALL  = "call"
    RAISE = "raise"
    ALLIN = "allin"


@dataclass
class Card:
    rank: str
    suit: str

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    @property
    def value(self):
        return RANK_VALUES[self.rank]


@dataclass
class PlayerState:
    player_id: str
    chips: int
    hole_cards: List[Card] = field(default_factory=list)
    is_folded: bool = False
    current_bet: int = 0
    is_allin: bool = False


class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for s in SUITS for r in RANKS]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> List[Card]:
        if len(self.cards) < n:
            raise ValueError("덱에 카드가 부족합니다")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt


class HandEvaluator:
    """
    핸드 강도 평가기
    정확한 포커 핸드 랭킹보단 근사치 스코어를 빠르게 뽑는 게 목적
    시뮬레이션에서 수천 번 호출되니 속도가 중요함
    """

    HAND_RANKS = {
        'high_card': 1, 'one_pair': 2, 'two_pair': 3,
        'three_of_a_kind': 4, 'straight': 5, 'flush': 6,
        'full_house': 7, 'four_of_a_kind': 8, 'straight_flush': 9
    }

    @staticmethod
    def evaluate(hole_cards: List[Card], community_cards: List[Card]) -> Tuple[int, float]:
        """
        Returns: (hand_rank, normalized_score 0~1)
        """
        all_cards = hole_cards + community_cards
        if len(all_cards) < 2:
            # 프리플랍 단계 - 홀카드 강도만
            return HandEvaluator._preflop_strength(hole_cards)

        hand_type = HandEvaluator._classify_hand(all_cards)
        rank = HandEvaluator.HAND_RANKS.get(hand_type, 1)
        # 0~1 정규화 (9등급 기준)
        score = rank / 9.0
        return rank, score

    @staticmethod
    def _preflop_strength(hole_cards: List[Card]) -> Tuple[int, float]:
        """프리플랍 핸드 강도 - 단순 룩업 테이블 방식"""
        if len(hole_cards) < 2:
            return 1, 0.1

        r1, r2 = hole_cards[0].value, hole_cards[1].value
        suited = hole_cards[0].suit == hole_cards[1].suit
        high = max(r1, r2)
        low = min(r1, r2)

        # AA, KK, QQ, AK suited 같은 프리미엄 핸드
        if r1 == r2 and r1 >= 12:   # 페어 Q+
            return 4, 0.85
        if r1 == r2 and r1 >= 8:    # 미들 페어
            return 3, 0.65
        if r1 == r2:                 # 로우 페어
            return 2, 0.45
        if high == 14 and low >= 10:  # A + 브로드웨이
            return 3, 0.70 if suited else 0.60
        if high >= 12 and low >= 10:
            return 3, 0.60 if suited else 0.50

        # 나머지는 두 카드 값 기반 선형 점수
        base = (high + low) / 28.0  # 최대 A+K = 27
        bonus = 0.05 if suited else 0.0
        return 1, min(base + bonus, 0.99)

    @staticmethod
    def _classify_hand(cards: List[Card]) -> str:
        """카드 리스트에서 가장 좋은 핸드 찾기"""
        from itertools import combinations

        best = 'high_card'
        best_rank = 1

        # 5장 조합 중 가장 좋은 거 찾기
        for combo in combinations(cards, min(5, len(cards))):
            ht = HandEvaluator._eval_five(list(combo))
            r = HandEvaluator.HAND_RANKS.get(ht, 1)
            if r > best_rank:
                best_rank = r
                best = ht

        return best

    @staticmethod
    def _eval_five(cards: List[Card]) -> str:
        values = sorted([c.value for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        val_counts = {}
        for v in values:
            val_counts[v] = val_counts.get(v, 0) + 1

        counts = sorted(val_counts.values(), reverse=True)
        is_flush = len(set(suits)) == 1
        is_straight = (len(set(values)) == 5 and values[0] - values[-1] == 4) or \
                      values == [14, 5, 4, 3, 2]  # A-2-3-4-5 wheel

        if is_flush and is_straight:
            return 'straight_flush'
        if counts[0] == 4:
            return 'four_of_a_kind'
        if counts[0] == 3 and counts[1] == 2:
            return 'full_house'
        if is_flush:
            return 'flush'
        if is_straight:
            return 'straight'
        if counts[0] == 3:
            return 'three_of_a_kind'
        if counts[0] == 2 and counts[1] == 2:
            return 'two_pair'
        if counts[0] == 2:
            return 'one_pair'
        return 'high_card'


@dataclass
class GameLog:
    """한 게임(핸드)의 로그"""
    game_id: str
    players: List[str]
    actions: List[dict]   # {player_id, street, action, amount, timestamp, hand_strength}
    pot_final: int
    winner: Optional[str]
    community_cards: List[str]


class PokerGame:
    """
    텍사스 홀덤 게임 1핸드를 진행하고 로그를 남기는 클래스

    블라인드 구조: SB = big_blind // 2, BB = big_blind
    """

    def __init__(self, player_ids: List[str], starting_chips: int = 10000, big_blind: int = 100):
        self.player_ids = player_ids
        self.starting_chips = starting_chips
        self.big_blind = big_blind
        self.small_blind = big_blind // 2

    def play_hand(self, game_id: str, players_obj: list, dealer_pos: int = 0) -> GameLog:
        """
        한 핸드 실행 후 GameLog 반환

        players_obj: Player 인스턴스 리스트 (player.py 에서 정의)
        """
        deck = Deck()
        community_cards = []
        pot = 0
        actions_log = []

        # 플레이어 상태 초기화
        states = {
            p.player_id: PlayerState(
                player_id=p.player_id,
                chips=self.starting_chips
            )
            for p in players_obj
        }

        # 홀카드 딜
        for p in players_obj:
            states[p.player_id].hole_cards = deck.deal(2)

        # 블라인드 포스트
        n = len(players_obj)
        sb_idx = (dealer_pos + 1) % n
        bb_idx = (dealer_pos + 2) % n
        sb_player = players_obj[sb_idx]
        bb_player = players_obj[bb_idx]

        states[sb_player.player_id].current_bet = self.small_blind
        states[bb_player.player_id].current_bet = self.big_blind
        pot += self.small_blind + self.big_blind

        # 거리별 액션 진행
        current_bet_to_call = self.big_blind
        for street in Street:
            if street == Street.FLOP:
                community_cards += deck.deal(3)
            elif street in (Street.TURN, Street.RIVER):
                community_cards += deck.deal(1)

            active_players = [p for p in players_obj
                              if not states[p.player_id].is_folded
                              and not states[p.player_id].is_allin]

            if len(active_players) <= 1:
                break

            for p in active_players:
                state = states[p.player_id]
                _, hand_score = HandEvaluator.evaluate(state.hole_cards, community_cards)

                # 팟 오즈 계산 (콜 결정에 쓰임)
                call_amount = current_bet_to_call - state.current_bet
                pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 0

                action, amount = p.decide_action(
                    street=street,
                    hand_strength=hand_score,
                    pot=pot,
                    call_amount=call_amount,
                    pot_odds=pot_odds,
                    current_bet=current_bet_to_call
                )

                # 액션 처리
                if action == Action.FOLD:
                    state.is_folded = True
                elif action == Action.CALL:
                    actual_call = min(call_amount, state.chips)
                    state.chips -= actual_call
                    state.current_bet += actual_call
                    pot += actual_call
                elif action == Action.RAISE:
                    raise_to = amount
                    actual_raise = min(raise_to - state.current_bet, state.chips)
                    state.chips -= actual_raise
                    state.current_bet += actual_raise
                    pot += actual_raise
                    current_bet_to_call = state.current_bet
                elif action == Action.ALLIN:
                    allin_amount = state.chips
                    pot += allin_amount
                    state.current_bet += allin_amount
                    state.chips = 0
                    state.is_allin = True
                    current_bet_to_call = max(current_bet_to_call, state.current_bet)

                import time
                actions_log.append({
                    'player_id': p.player_id,
                    'street': street.value,
                    'action': action.value,
                    'amount': amount,
                    'pot_at_action': pot,
                    'hand_strength': round(hand_score, 4),
                    'pot_odds': round(pot_odds, 4),
                    'timestamp': time.time(),
                })

            current_bet_to_call = 0
            for st in states.values():
                st.current_bet = 0

        # 승자 결정 (간소화 - 남은 플레이어 중 핸드 강도 최대)
        remaining = [
            p for p in players_obj
            if not states[p.player_id].is_folded
        ]

        if len(remaining) == 1:
            winner = remaining[0].player_id
        elif remaining:
            winner = max(
                remaining,
                key=lambda p: HandEvaluator.evaluate(
                    states[p.player_id].hole_cards, community_cards
                )[1]
            ).player_id
        else:
            winner = None

        return GameLog(
            game_id=game_id,
            players=[p.player_id for p in players_obj],
            actions=actions_log,
            pot_final=pot,
            winner=winner,
            community_cards=[str(c) for c in community_cards]
        )
