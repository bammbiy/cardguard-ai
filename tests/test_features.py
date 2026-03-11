"""
피처 엔지니어링 테스트

피처 추출이 제대로 되는지, 값 범위가 말이 되는지 확인
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.detection.feature_engineering import (
    compute_aggregate_features,
    encode_action,
    build_sequences,
    SEQ_LENGTH,
)
from src.simulator.poker_game import Street


def _make_action(action='call', street='flop', hand_strength=0.5, pot_odds=0.3, amount=100, pot=400):
    return {
        'player_id': 'p_test',
        'action': action,
        'street': street,
        'hand_strength': hand_strength,
        'pot_odds': pot_odds,
        'amount': amount,
        'pot_at_action': pot,
        'timestamp': 0.0,
    }


class TestAggregateFeatures:
    def test_empty_actions_returns_empty(self):
        result = compute_aggregate_features([])
        assert result == {}

    def test_fold_rate_correct(self):
        actions = [_make_action('fold')] * 3 + [_make_action('call')] * 7
        feats = compute_aggregate_features(actions)
        assert abs(feats['fold_rate'] - 0.3) < 1e-6

    def test_all_rates_sum_to_one(self):
        actions = [
            _make_action('fold'), _make_action('call'),
            _make_action('raise'), _make_action('check'), _make_action('allin')
        ]
        feats = compute_aggregate_features(actions)
        total = feats['fold_rate'] + feats['call_rate'] + feats['raise_rate'] + \
                feats['check_rate'] + feats['allin_rate']
        assert abs(total - 1.0) < 1e-6

    def test_feature_values_in_valid_range(self):
        actions = [_make_action(a) for a in ['fold', 'call', 'raise', 'check', 'call', 'fold']]
        feats = compute_aggregate_features(actions)
        for key in ['fold_rate', 'call_rate', 'raise_rate', 'check_rate', 'allin_rate']:
            assert 0.0 <= feats[key] <= 1.0, f"{key}={feats[key]} 범위 벗어남"

    def test_strong_fold_rate_with_all_folds(self):
        """강한 핸드에서 전부 폴드하면 strong_fold_rate = 1.0"""
        actions = [_make_action('fold', hand_strength=0.85) for _ in range(10)]
        feats = compute_aggregate_features(actions)
        assert feats['strong_fold_rate'] == 1.0


class TestEncodeAction:
    def test_output_dimension(self):
        action = _make_action()
        vec = encode_action(action)
        assert len(vec) == 12

    def test_action_onehot_is_valid(self):
        for a in ['fold', 'check', 'call', 'raise', 'allin']:
            vec = encode_action(_make_action(action=a))
            # action one-hot (첫 5개) 중 1이 하나만 있어야
            assert sum(vec[:5]) == 1.0

    def test_hand_strength_preserved(self):
        vec = encode_action(_make_action(hand_strength=0.77))
        assert abs(vec[9] - 0.77) < 1e-6

    def test_raise_ratio_clipped(self):
        # amount가 pot보다 훨씬 크면 클리핑 되어야
        vec = encode_action(_make_action(action='raise', amount=100000, pot=100))
        assert vec[11] <= 5.0


class TestBuildSequences:
    def test_returns_empty_if_too_short(self):
        actions = [_make_action() for _ in range(SEQ_LENGTH - 1)]
        result = build_sequences(actions)
        assert result.size == 0

    def test_shape_correct(self):
        n = SEQ_LENGTH + 5
        actions = [_make_action() for _ in range(n)]
        seqs = build_sequences(actions, seq_len=SEQ_LENGTH)
        # 슬라이딩 윈도우: n - seq_len + 1 개
        expected_n = n - SEQ_LENGTH + 1
        assert seqs.shape == (expected_n, SEQ_LENGTH, 12)

    def test_values_in_range(self):
        actions = [_make_action() for _ in range(SEQ_LENGTH + 10)]
        seqs = build_sequences(actions)
        assert seqs.min() >= 0.0
        # one-hot이니까 최대값은 1.0 이상이면 안 됨 (raise ratio 제외하면)
        # raise ratio는 최대 5.0으로 클리핑됨
        assert seqs.max() <= 5.01
