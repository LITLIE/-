from __future__ import annotations

from typing import Callable, Dict, Optional


class AppStateMachine:
    """
    有限状态机：
    IDLE -> ACTIVE -> OPERATING -> COOLDOWN
    设计原则：
    - IDLE：无有效交互，或者长时间未检测到手
    - ACTIVE：检测到手且允许响应手势
    - OPERATING：发生一次性动作
    - COOLDOWN：动作冷却，防止重复触发
    """

    def __init__(
        self,
        cooldown_frames: int = 8,
        idle_lost_frames: int = 8,
        activate_hold_frames: int = 6,
        confirm_hold_frames: int = 3,
        deactivate_hold_frames: int = 8,
    ):
        self.STATES = ['IDLE', 'ACTIVE', 'OPERATING', 'COOLDOWN']
        self.current_state = 'IDLE'

        self.gesture_hold_count = 0
        self.last_gesture = '无'

        self.no_hand_count = 0
        self.cooldown_counter = 0

        self.COOLDOWN_FRAMES = int(cooldown_frames)
        self.IDLE_LOST_FRAMES = int(idle_lost_frames)
        self.ACTIVATE_HOLD_FRAMES = int(activate_hold_frames)
        self.CONFIRM_HOLD_FRAMES = int(confirm_hold_frames)
        self.DEACTIVATE_HOLD_FRAMES = int(deactivate_hold_frames)

        self._resume_state = 'ACTIVE'
        self.ai_busy = False

        self.callbacks: Dict[str, Optional[Callable]] = {
            'on_activate': None,
            'on_deactivate': None,
            'on_swipe': None,
            'on_confirm': None,
            'on_multimodal_action': None,
            'on_state_change': None,
        }

    def bind_callback(self, event_name, func):
        if event_name in self.callbacks:
            self.callbacks[event_name] = func

    def _trigger(self, event_name, *args):
        cb = self.callbacks.get(event_name)
        if cb:
            cb(*args)

    def _set_state(self, new_state: str):
        if new_state not in self.STATES:
            return
        if self.current_state != new_state:
            self.current_state = new_state
            self._trigger('on_state_change', new_state)

    def _enter_cooldown(self, resume_state: str = 'ACTIVE'):
        self._resume_state = resume_state if resume_state in self.STATES else 'ACTIVE'
        self.current_state = 'COOLDOWN'
        self.cooldown_counter = self.COOLDOWN_FRAMES
        self.gesture_hold_count = 0
        self._trigger('on_state_change', 'COOLDOWN')

    def _is_confirm(self, gesture: str) -> bool:
        return gesture in {'Close', '握拳'}

    def _is_activate(self, gesture: str) -> bool:
        return gesture in {'OK'}

    def _is_deactivate(self, gesture: str) -> bool:
        return gesture in {'Open', '五指张开'}

    def _is_swipe(self, gesture: str) -> bool:
        return gesture in {'左滑', '右滑', '向左滑动', '向右滑动'}

    def reset(self):
        self.current_state = 'IDLE'
        self.gesture_hold_count = 0
        self.last_gesture = '无'
        self.no_hand_count = 0
        self.cooldown_counter = 0
        self._resume_state = 'ACTIVE'
        self.ai_busy = False
        self._trigger('on_state_change', 'IDLE')

    def update(
        self,
        static_gesture: str,
        motion_gesture: str,
        voice_command: Optional[str] = None,
        hand_count: int = 0,
        ai_busy: bool = False,
    ):
        """
        返回当前状态。
        关键约束：
        - voice_command 必须是“一次性事件”，上层只在真正有新语音时传入一次
        - hand_count 用于判定是否进入 IDLE
        """
        self.ai_busy = bool(ai_busy)

        # 没有手时，累计空手帧；超过阈值后回到 IDLE
        if hand_count <= 0:
            self.no_hand_count += 1
        else:
            self.no_hand_count = 0

        if self.no_hand_count >= self.IDLE_LOST_FRAMES and self.current_state != 'IDLE':
            self._set_state('IDLE')
            self.gesture_hold_count = 0
            self.last_gesture = '无'
            return self.current_state

        # 冷却阶段只做倒计时，不重复响应
        if self.current_state == 'COOLDOWN':
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.cooldown_counter = 0
                self._set_state(self._resume_state if self._resume_state in self.STATES else 'ACTIVE')
            return self.current_state

        # AI 正在处理时，不再接收新的交互触发
        if self.ai_busy:
            return self.current_state

        # 手势稳定计数
        if static_gesture == self.last_gesture and static_gesture != '无':
            self.gesture_hold_count += 1
        else:
            self.gesture_hold_count = 0
            self.last_gesture = static_gesture

        # 语音触发：上层必须保证它只传一次
        if voice_command:
            self._set_state('OPERATING')
            self._trigger('on_multimodal_action', voice_command, static_gesture)
            self._enter_cooldown('ACTIVE')
            return self.current_state

        if self.current_state == 'IDLE':
            if self.gesture_hold_count >= self.ACTIVATE_HOLD_FRAMES and self._is_activate(static_gesture):
                self._set_state('ACTIVE')
                self._trigger('on_activate')
                self.gesture_hold_count = 0

        elif self.current_state == 'ACTIVE':
            if motion_gesture != '无' and self._is_swipe(motion_gesture):
                self._set_state('OPERATING')
                self._trigger('on_swipe', motion_gesture)
                self._enter_cooldown('ACTIVE')

            elif self.gesture_hold_count >= self.CONFIRM_HOLD_FRAMES and self._is_confirm(static_gesture):
                self._set_state('OPERATING')
                self._trigger('on_confirm')
                self._enter_cooldown('ACTIVE')

            elif self.gesture_hold_count >= self.DEACTIVATE_HOLD_FRAMES and self._is_deactivate(static_gesture):
                self._set_state('IDLE')
                self._trigger('on_deactivate')
                self.gesture_hold_count = 0

        return self.current_state