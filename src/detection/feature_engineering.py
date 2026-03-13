"""
피처 엔지니어링

게임 로그 → 플레이어별 통계/시계열 피처로 변환
탐지 모델이 직접 쓰는 입력값들

피처 설계할 때 가장 고민한 부분:
- 단순 집계 통계 (폴드율, 레이즈율 등)은 IF에
- 순서/패턴 정보가 담긴 시퀀스는 LSTM에
두 가지를 분리해서 처리
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


# LSTM에 쓸 시퀀스 길이
SEQ_LENGTH = 20


def load_logs(log_path: str) -> Tuple[list, dict]:
    """게임 로그 파일 로드"""
    with open(log_path, 'r', encoding='utf-8') as f:
        sessions = json.load(f)

    label_path = Path(log_path).parent / 'player_labels.json'
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    return sessions, labels


def extract_player_actions(sessions: list) -> Dict[str, List[dict]]:
    """
    플레이어 ID별로 전체 액션 리스트 모으기
    세션/핸드 구분 없이 시간 순서대로
    """
    player_actions = {}

    for session in sessions:
        for hand in session.get('hands', []):
            for action in hand.get('actions', []):
                pid = action['player_id']
                if pid not in player_actions:
                    player_actions[pid] = []
                player_actions[pid].append(action)

    return player_actions


# ─────────────────────────────────────────
#  집계 피처 (Isolation Forest용)
# ─────────────────────────────────────────

def compute_aggregate_features(actions: List[dict]) -> dict:
    """
    플레이어의 전체 액션 리스트 → 집계 통계 피처
    """
    if not actions:
        return {}

    total = len(actions)
    action_types = [a['action'] for a in actions]
    hand_strengths = [a['hand_strength'] for a in actions]
    pot_odds_list = [a['pot_odds'] for a in actions]
    amounts = [a.get('amount', 0) or 0 for a in actions]
    pots = [a.get('pot_at_action', 1) for a in actions]

    # 기본 액션 비율
    fold_rate = action_types.count('fold') / total
    call_rate = action_types.count('call') / total
    raise_rate = action_types.count('raise') / total
    check_rate = action_types.count('check') / total
    allin_rate = action_types.count('allin') / total

    # 핸드 강도 통계
    hs_mean = np.mean(hand_strengths)
    hs_std = np.std(hand_strengths)

    # 핵심: 핸드 강도 대비 행동 일관성
    # 정상 플레이어는 강한 핸드에서 레이즈, 약한 핸드에서 폴드 경향
    # 어뷰저는 이게 반대거나 랜덤함
    strong_hand_mask = [hs > 0.6 for hs in hand_strengths]
    weak_hand_mask = [hs < 0.35 for hs in hand_strengths]

    strong_fold_rate = _conditional_rate(action_types, strong_hand_mask, 'fold')
    strong_raise_rate = _conditional_rate(action_types, strong_hand_mask, 'raise')
    weak_fold_rate = _conditional_rate(action_types, weak_hand_mask, 'fold')
    weak_call_rate = _conditional_rate(action_types, weak_hand_mask, 'call')

    # 팟 오즈 대비 콜 패턴
    # 정상: 팟 오즈가 유리할 때 콜 / 어뷰저(overcall 타입): 팟 오즈 무시하고 콜
    call_mask = [a == 'call' for a in action_types]
    call_pot_odds = [po for po, m in zip(pot_odds_list, call_mask) if m]
    avg_call_pot_odds = np.mean(call_pot_odds) if call_pot_odds else 0.0

    # 레이즈 사이징의 일관성 (봇은 이게 너무 규칙적)
    raise_mask = [a == 'raise' for a in action_types]
    raise_amounts = [amt for amt, m in zip(amounts, raise_mask) if m]
    raise_pots = [p for p, m in zip(pots, raise_mask) if m]
    if raise_pots and raise_amounts:
        raise_ratios = [a / max(p, 1) for a, p in zip(raise_amounts, raise_pots)]
        raise_ratio_std = np.std(raise_ratios)    # 봇은 이 값이 매우 작음
        raise_ratio_mean = np.mean(raise_ratios)
    else:
        raise_ratio_std = 0.0
        raise_ratio_mean = 0.0

    # 핸드 강도-팟 오즈 상관관계
    # 정상 플레이어는 약한 핸드 + 높은 팟 오즈 = 폴드 경향 보여야 함
    hs_po_corr = np.corrcoef(hand_strengths, pot_odds_list)[0, 1] if len(actions) > 2 else 0.0

    # 올인 빈도 × 평균 핸드 강도 (칩 덤핑 탐지)
    allin_hs = [hs for hs, a in zip(hand_strengths, action_types) if a == 'allin']
    avg_allin_hs = np.mean(allin_hs) if allin_hs else 0.0

    return {
        # 기본 행동 비율
        'fold_rate': fold_rate,
        'call_rate': call_rate,
        'raise_rate': raise_rate,
        'check_rate': check_rate,
        'allin_rate': allin_rate,

        # 핸드 강도 통계
        'hs_mean': hs_mean,
        'hs_std': hs_std,

        # 강도 대비 행동 일관성 (핵심 피처)
        'strong_fold_rate': strong_fold_rate,
        'strong_raise_rate': strong_raise_rate,
        'weak_fold_rate': weak_fold_rate,
        'weak_call_rate': weak_call_rate,

        # 팟 오즈 관련
        'avg_call_pot_odds': avg_call_pot_odds,
        'hs_po_corr': hs_po_corr if not np.isnan(hs_po_corr) else 0.0,

        # 레이즈 패턴
        'raise_ratio_mean': raise_ratio_mean,
        'raise_ratio_std': raise_ratio_std,   # 봇 탐지용

        # 올인 관련
        'avg_allin_hs': avg_allin_hs,
        'allin_count': len(allin_hs),
    }


def _conditional_rate(action_types: list, mask: list, target_action: str) -> float:
    """마스크 조건 하에서 특정 액션 비율"""
    masked = [a for a, m in zip(action_types, mask) if m]
    if not masked:
        return 0.0
    return masked.count(target_action) / len(masked)


# ─────────────────────────────────────────
#  시퀀스 피처 (LSTM용)
# ─────────────────────────────────────────

ACTION_ENCODING = {
    'fold': 0, 'check': 1, 'call': 2, 'raise': 3, 'allin': 4
}
STREET_ENCODING = {
    'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3
}


def encode_action(action: dict) -> List[float]:
    """
    단일 액션 → 수치 벡터
    [action_onehot(5), street_onehot(4), hand_strength, pot_odds, raise_ratio]
    총 12차원
    """
    vec = [0.0] * 12

    # action one-hot
    a_idx = ACTION_ENCODING.get(action['action'], 0)
    vec[a_idx] = 1.0

    # street one-hot
    s_idx = STREET_ENCODING.get(action.get('street', 'preflop'), 0)
    vec[5 + s_idx] = 1.0

    # 연속형 피처
    vec[9] = float(action.get('hand_strength', 0.0))
    vec[10] = float(action.get('pot_odds', 0.0))

    # raise ratio (raise면 amount/pot, 아니면 0)
    if action['action'] in ('raise', 'allin'):
        pot = action.get('pot_at_action', 1)
        amt = action.get('amount', 0) or 0
        vec[11] = min(amt / max(pot, 1), 5.0)   # 5로 클리핑

    return vec


def build_sequences(actions: List[dict], seq_len: int = SEQ_LENGTH) -> np.ndarray:
    """
    액션 리스트 → 슬라이딩 윈도우 시퀀스 배열
    Returns: shape (n_sequences, seq_len, 12)
    """
    if len(actions) < seq_len:
        return np.array([])

    encoded = np.array([encode_action(a) for a in actions])

    sequences = []
    for i in range(len(encoded) - seq_len + 1):
        sequences.append(encoded[i: i + seq_len])

    return np.array(sequences)


# ─────────────────────────────────────────
#  전체 파이프라인
# ─────────────────────────────────────────

def build_dataset(sessions: list, labels: dict):
    """
    세션 리스트 → (집계 피처 DataFrame, 시퀀스 배열, 레이블)
    """
    player_actions = extract_player_actions(sessions)

    agg_rows = []
    seq_list = []
    seq_labels = []

    for pid, actions in player_actions.items():
        if len(actions) < SEQ_LENGTH:
            continue   # 데이터 너무 적은 플레이어 스킵

        label = labels.get(pid, 'normal')
        is_abnormal = 0 if label == 'normal' else 1

        # 집계 피처
        agg_feat = compute_aggregate_features(actions)
        if agg_feat:
            agg_feat['player_id'] = pid
            agg_feat['label'] = is_abnormal
            agg_rows.append(agg_feat)

        # 시퀀스 피처
        seqs = build_sequences(actions)
        if seqs.size > 0:
            seq_list.append(seqs)
            seq_labels.extend([is_abnormal] * len(seqs))

    agg_df = pd.DataFrame(agg_rows)

    if seq_list:
        seq_array = np.vstack(seq_list)
        seq_label_array = np.array(seq_labels)
    else:
        seq_array = np.array([])
        seq_label_array = np.array([])

    return agg_df, seq_array, seq_label_array


def save_dataset(agg_df: pd.DataFrame, seq_array: np.ndarray, seq_labels: np.ndarray, output_dir: str):
    """처리된 피처 저장"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    agg_df.to_csv(out / 'aggregate_features.csv', index=False)
    if seq_array.size > 0:
        np.save(out / 'sequences.npy', seq_array)
        np.save(out / 'seq_labels.npy', seq_labels)

    print(f"피처 저장 완료 → {out}")
    print(f"  집계 피처: {agg_df.shape}")
    print(f"  시퀀스: {seq_array.shape}")
    print(f"  이상 비율: {seq_labels.mean():.1%}" if seq_labels.size > 0 else "")
