"""
플레이어 행동 모델

정상 플레이어 / 어뷰저 / 봇 세 가지 타입을 구현
각 타입이 어떻게 행동이 다른지가 탐지 모델의 학습 타겟

실제 어뷰저 행동을 최대한 현실적으로 모사하려고 했는데
완벽하진 않고 계속 개선이 필요한 부분
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Tuple

from .poker_game import Action, Street


class BasePlayer(ABC):
    def __init__(self, player_id: str, player_type: str):
        self.player_id = player_id
        self.player_type = player_type  # 'normal', 'abuser', 'bot'

    @abstractmethod
    def decide_action(
        self,
        street: Street,
        hand_strength: float,
        pot: int,
        call_amount: int,
        pot_odds: float,
        current_bet: int,
    ) -> Tuple[Action, int]:
        pass

    def _think_delay(self, base: float = 1.5, variance: float = 2.0) -> float:
        """
        의사결정 딜레이 시뮬레이션
        실제 사람은 생각하는 시간이 상황마다 다름
        봇이랑 구분하는 핵심 피처 중 하나
        """
        return max(0.3, random.gauss(base, variance))


class NormalPlayer(BasePlayer):
    """
    일반 플레이어 - GTO와 직관 사이 어딘가
    항상 최선을 다하진 않고 가끔 블러프도 하고 실수도 함
    """

    def __init__(self, player_id: str, aggression: float = 0.5, bluff_rate: float = 0.15):
        super().__init__(player_id, 'normal')
        self.aggression = aggression        # 0 = 패시브, 1 = 어그레시브
        self.bluff_rate = bluff_rate

    def decide_action(self, street, hand_strength, pot, call_amount, pot_odds, current_bet) -> Tuple[Action, int]:
        delay = self._think_delay(base=2.0, variance=2.5)   # 평균 2초, 꽤 분산 있음
        time.sleep(0)   # 실제론 delay 쓰지만 시뮬에선 생략 (나중에 로그에 기록만)

        # 블러프 확률
        is_bluffing = random.random() < self.bluff_rate

        effective_strength = hand_strength
        if is_bluffing:
            effective_strength = min(1.0, hand_strength + random.uniform(0.2, 0.5))

        # 기본 의사결정 로직
        if effective_strength < 0.25:
            # 약한 핸드 - 대부분 폴드, 가끔 블러프 레이즈
            if call_amount > 0 and random.random() > 0.2:
                return Action.FOLD, 0
            if call_amount == 0:
                return Action.CHECK, 0
            return Action.CALL, call_amount

        elif effective_strength < 0.55:
            # 중간 핸드 - 팟 오즈 보고 결정
            if call_amount == 0:
                if random.random() < self.aggression * 0.4:
                    raise_size = int(pot * random.uniform(0.4, 0.7))
                    return Action.RAISE, max(raise_size, current_bet * 2)
                return Action.CHECK, 0
            if pot_odds < effective_strength * 0.8:
                return Action.CALL, call_amount
            return Action.FOLD, 0

        else:
            # 강한 핸드 - 밸류 뽑기
            if random.random() < self.aggression * 0.7:
                raise_size = int(pot * random.uniform(0.6, 1.2))
                # 가끔 올인
                if hand_strength > 0.85 and random.random() < 0.25:
                    return Action.ALLIN, 0
                return Action.RAISE, max(raise_size, current_bet * 2)
            if call_amount > 0:
                return Action.CALL, call_amount
            return Action.CHECK, 0


class AbuserPlayer(BasePlayer):
    """
    어뷰저 플레이어 - 칩 덤핑 / 콜루전 유형

    특징:
    - 강한 핸드에서 오히려 체크/폴드 (의도적 패배)
    - 특정 플레이어한테만 과하게 콜
    - 베팅 사이즈가 불규칙 (팟 대비 말이 안 되는 금액)
    - 폴드 타이밍이 비정상적으로 빠름
    """

    def __init__(self, player_id: str, collude_partner: str = None, abuse_type: str = 'chip_dump'):
        super().__init__(player_id, 'abuser')
        self.collude_partner = collude_partner
        self.abuse_type = abuse_type   # 'chip_dump', 'collusion', 'overcall'
        self._action_count = 0

    def decide_action(self, street, hand_strength, pot, call_amount, pot_odds, current_bet) -> Tuple[Action, int]:
        self._action_count += 1

        # 칩덤핑: 핸드 강도랑 반대로 행동
        if self.abuse_type == 'chip_dump':
            return self._chip_dump_action(hand_strength, pot, call_amount, current_bet)

        elif self.abuse_type == 'collusion':
            return self._collusion_action(hand_strength, pot, call_amount, current_bet)

        elif self.abuse_type == 'overcall':
            return self._overcall_action(hand_strength, pot, call_amount, current_bet)

        # fallback
        if call_amount > 0:
            return Action.CALL, call_amount
        return Action.CHECK, 0

    def _chip_dump_action(self, hand_strength, pot, call_amount, current_bet):
        """좋은 핸드일수록 일부러 약하게 플레이"""
        # 강한 핸드 → 폴드하거나 최소 콜만
        if hand_strength > 0.7:
            if call_amount > 0 and random.random() < 0.55:
                return Action.FOLD, 0
            if call_amount > 0:
                return Action.CALL, call_amount
            return Action.CHECK, 0

        # 약한 핸드 → 오히려 큰 레이즈
        if hand_strength < 0.3:
            if random.random() < 0.6:
                raise_size = int(pot * random.uniform(1.5, 3.0))   # 팟 대비 말이 안 되는 크기
                return Action.RAISE, max(raise_size, current_bet * 3)
            if random.random() < 0.3:
                return Action.ALLIN, 0

        if call_amount > 0:
            return Action.CALL, call_amount
        return Action.CHECK, 0

    def _collusion_action(self, hand_strength, pot, call_amount, current_bet):
        """파트너 있을 때 패시브하게 플레이해서 파트너 이기도록 도움"""
        # 파트너가 있으면 강한 핸드에서도 체크/콜만
        if hand_strength > 0.6:
            if call_amount > 0:
                return Action.CALL, call_amount
            return Action.CHECK, 0

        # 약할 때 빠른 폴드 (팟에 기여 최소화)
        if call_amount > 0:
            if random.random() < 0.75:
                return Action.FOLD, 0
            return Action.CALL, call_amount
        return Action.CHECK, 0

    def _overcall_action(self, hand_strength, pot, call_amount, current_bet):
        """팟 오즈 무시하고 지나치게 많이 콜"""
        # 핸드 강도 상관없이 거의 항상 콜
        if call_amount > 0:
            if random.random() < 0.85:   # 85% 확률로 콜 (정상이라면 절대 이럴 수 없음)
                return Action.CALL, call_amount
        if hand_strength > 0.5 and random.random() < 0.3:
            raise_size = int(pot * random.uniform(0.3, 0.6))
            return Action.RAISE, max(raise_size, current_bet * 2)
        return Action.CHECK, 0


class BotPlayer(BasePlayer):
    """
    봇 플레이어 - AI가 플레이하는 척하는 유형

    봇의 특징:
    - 반응 시간이 일정하거나 매우 짧음 (분산이 극히 작음)
    - 베팅 패턴이 기계적으로 반복됨
    - GTO에 가까운 의사결정 (너무 잘함)
    - 리버 블러프 비율이 수학적으로 정확함
    """

    def __init__(self, player_id: str, skill_level: float = 0.85):
        super().__init__(player_id, 'bot')
        self.skill_level = skill_level
        self._prev_actions = []

    def decide_action(self, street, hand_strength, pot, call_amount, pot_odds, current_bet) -> Tuple[Action, int]:
        # 봇은 생각 시간이 거의 일정 (0.3~0.8초, 분산 매우 작음 - 핵심 피처!)
        delay = random.gauss(0.5, 0.08)
        # time.sleep(delay)  # 시뮬에선 생략

        # GTO 기반 의사결정
        # EV = hand_strength - pot_odds 라고 단순화
        if call_amount > 0:
            ev = hand_strength - pot_odds
            if ev > 0.1:
                # 밸류 레이즈
                if hand_strength > 0.7 and random.random() < 0.6:
                    raise_size = self._gto_raise_size(pot, hand_strength)
                    return Action.RAISE, raise_size
                return Action.CALL, call_amount
            elif ev > -0.05:
                # 마지널 - 확률적으로 콜/폴드
                if random.random() < (ev + 0.05) / 0.15:
                    return Action.CALL, call_amount
                return Action.FOLD, 0
            else:
                # -EV는 폴드 (블러프 캐칭 비율만큼 콜)
                bluff_catch_rate = 0.15   # 항상 정확히 15% - 이 패턴이 비자연스러움
                if random.random() < bluff_catch_rate:
                    return Action.CALL, call_amount
                return Action.FOLD, 0
        else:
            # 체크 or 베팅
            if hand_strength > 0.5:
                raise_size = self._gto_raise_size(pot, hand_strength)
                return Action.RAISE, raise_size
            # 블러프 빈도도 수학적으로 정확
            if street == Street.RIVER and random.random() < 0.33:
                raise_size = int(pot * 0.75)
                return Action.RAISE, raise_size
            return Action.CHECK, 0

    def _gto_raise_size(self, pot: int, hand_strength: float) -> int:
        """GTO 레이즈 사이징 - 봇은 이게 너무 일정함"""
        # 강도별로 정해진 팟 비율 (사람은 이렇게 기계적이지 않음)
        if hand_strength > 0.85:
            ratio = 0.75
        elif hand_strength > 0.65:
            ratio = 0.50
        else:
            ratio = 0.33
        return int(pot * ratio)
