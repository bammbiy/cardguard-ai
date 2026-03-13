"""
시뮬레이터 관련 테스트

게임 엔진이 룰대로 돌아가는지 기본적인 거 체크
버그 찾다가 추가한 케이스들도 섞여 있음
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.simulator.poker_game import (
    Card, Deck, HandEvaluator, PokerGame, Action, Street
)
from src.simulator.player import NormalPlayer, AbuserPlayer, BotPlayer


# ─── Card / Deck ───────────────────────────────────────

class TestDeck:
    def test_deck_has_52_cards(self):
        deck = Deck()
        assert len(deck.cards) == 52

    def test_no_duplicate_cards(self):
        deck = Deck()
        card_strs = [str(c) for c in deck.cards]
        assert len(card_strs) == len(set(card_strs))

    def test_deal_reduces_deck(self):
        deck = Deck()
        dealt = deck.deal(5)
        assert len(dealt) == 5
        assert len(deck.cards) == 47

    def test_deal_too_many_raises(self):
        deck = Deck()
        with pytest.raises(ValueError):
            deck.deal(53)


# ─── HandEvaluator ─────────────────────────────────────

class TestHandEvaluator:
    def _make_cards(self, specs):
        """'A♠' 같은 문자열 리스트 → Card 리스트"""
        cards = []
        for spec in specs:
            rank = spec[:-1]
            suit = spec[-1]
            cards.append(Card(rank, suit))
        return cards

    def test_royal_flush_is_straight_flush(self):
        hole = self._make_cards(['A♠', 'K♠'])
        community = self._make_cards(['Q♠', 'J♠', '10♠'])
        rank, score = HandEvaluator.evaluate(hole, community)
        assert rank == 9   # straight_flush
        assert score > 0.9

    def test_four_of_a_kind(self):
        hole = self._make_cards(['A♠', 'A♥'])
        community = self._make_cards(['A♦', 'A♣', '2♠'])
        rank, score = HandEvaluator.evaluate(hole, community)
        assert rank == 8

    def test_full_house(self):
        hole = self._make_cards(['K♠', 'K♥'])
        community = self._make_cards(['K♦', '3♣', '3♠'])
        rank, score = HandEvaluator.evaluate(hole, community)
        assert rank == 7

    def test_preflop_pocket_aces(self):
        hole = self._make_cards(['A♠', 'A♥'])
        rank, score = HandEvaluator.evaluate(hole, [])
        assert score >= 0.8   # 포켓 에이스는 강해야

    def test_preflop_72_offsuit(self):
        hole = self._make_cards(['7♠', '2♥'])
        _, score = HandEvaluator.evaluate(hole, [])
        assert score < 0.35   # 72오프는 약해야


# ─── Player ─────────────────────────────────────────────

class TestPlayers:
    def test_normal_player_returns_valid_action(self):
        player = NormalPlayer('p1', aggression=0.5)
        action, amount = player.decide_action(
            street=Street.PREFLOP,
            hand_strength=0.6,
            pot=200,
            call_amount=100,
            pot_odds=0.33,
            current_bet=100,
        )
        assert action in list(Action)
        assert amount >= 0

    def test_abuser_chip_dump_raises_with_weak_hand(self):
        """칩 덤핑 어뷰저는 약한 핸드에서 레이즈 많이 해야 함"""
        player = AbuserPlayer('p_abuse', abuse_type='chip_dump')
        raise_count = 0
        n_trials = 100

        for _ in range(n_trials):
            action, _ = player.decide_action(
                street=Street.FLOP,
                hand_strength=0.15,   # 매우 약한 핸드
                pot=500,
                call_amount=0,
                pot_odds=0.0,
                current_bet=0,
            )
            if action == Action.RAISE:
                raise_count += 1

        # 정상 플레이어라면 약한 핸드에서 레이즈를 거의 안 해야 하는데
        # 칩 덤핑 어뷰저는 많이 함 (60% 이상)
        assert raise_count / n_trials > 0.5, f"레이즈 비율: {raise_count/n_trials:.2f}"

    def test_bot_has_consistent_raise_sizing(self):
        """봇은 레이즈 사이징 분산이 작아야 함 (핵심 탐지 피처)"""
        import numpy as np
        player = BotPlayer('bot1', skill_level=0.85)
        ratios = []

        for _ in range(50):
            action, amount = player.decide_action(
                street=Street.FLOP,
                hand_strength=0.75,
                pot=400,
                call_amount=0,
                pot_odds=0.0,
                current_bet=0,
            )
            if action == Action.RAISE and amount > 0:
                ratios.append(amount / 400)

        if ratios:
            std = np.std(ratios)
            assert std < 0.2, f"봇 레이즈 비율 표준편차가 너무 큼: {std:.3f}"


# ─── PokerGame ──────────────────────────────────────────

class TestPokerGame:
    def _make_players(self, n=4):
        return [NormalPlayer(f'p{i}', aggression=0.5) for i in range(n)]

    def test_game_runs_without_error(self):
        players = self._make_players(4)
        game = PokerGame([p.player_id for p in players], starting_chips=5000, big_blind=100)
        log = game.play_hand('test_hand_001', players, dealer_pos=0)
        assert log is not None
        assert log.game_id == 'test_hand_001'

    def test_log_has_actions(self):
        players = self._make_players(4)
        game = PokerGame([p.player_id for p in players], starting_chips=5000, big_blind=100)
        log = game.play_hand('test_hand_002', players, dealer_pos=0)
        assert len(log.actions) > 0

    def test_winner_is_valid_player(self):
        players = self._make_players(4)
        game = PokerGame([p.player_id for p in players], starting_chips=5000, big_blind=100)
        log = game.play_hand('test_hand_003', players, dealer_pos=0)
        valid_ids = {p.player_id for p in players}
        assert log.winner in valid_ids or log.winner is None

    def test_pot_is_positive(self):
        players = self._make_players(4)
        game = PokerGame([p.player_id for p in players], starting_chips=5000, big_blind=100)
        log = game.play_hand('test_hand_004', players, dealer_pos=0)
        # 블라인드 최소 150 (SB50 + BB100)
        assert log.pot_final >= 150
