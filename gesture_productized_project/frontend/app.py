from __future__ import annotations

import os
import platform
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple
import pyvirtualcam
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from frontend.animations import AnimationManager, HeartBurst, RingPulse, SparkParticle

THIS_FILE = Path(__file__).resolve()
FRONTEND_DIR = THIS_FILE.parent
PROJECT_ROOT = FRONTEND_DIR.parent
for p in (PROJECT_ROOT, FRONTEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CHINESE_FONT_PATH = 'C:/Windows/Fonts/simhei.ttf'#中文字体
if not os.path.exists(CHINESE_FONT_PATH):
    CHINESE_FONT_PATH = 'simhei.ttf'

from config.settings import Config, SCENES
from backend.gesture_module.engine import FrameObservation, GestureEngine
from backend.gesture_module.state_machine import AppStateMachine
from backend.llm_module.ai_service import AIService
from backend.speech_module.voice_service import VoiceService


def put_chinese_text(frame, text, position, font_size=20, color=(255, 255, 255)):
    if not os.path.exists(CHINESE_FONT_PATH):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return frame
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(CHINESE_FONT_PATH, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class GlowCard(QFrame):
    def __init__(self, accent: str = '#38BDF8', radius: int = 16, fixed_width: Optional[int] = None):
        super().__init__()
        self.setObjectName('GlowCard')
        if fixed_width:
            self.setFixedWidth(fixed_width)
        self.setStyleSheet(f'''
            QFrame#GlowCard {{
                background-color: rgba(15, 23, 42, 230);
                border: 1px solid rgba(148, 163, 184, 50);
                border-radius: {radius}px;
            }}
            QLabel {{ color: #E2E8F0; background: transparent; }}
            QPushButton {{ background: transparent; }}
            QListWidget {{ background: transparent; border: none; }}
        ''')
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 0)
        shadow.setColor(QColor(accent))
        self.setGraphicsEffect(shadow)


class TouchlessAssistantApp(QWidget):
    frame_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle('NO-TOUCH AI INTERFACE')
        self.setMinimumSize(1600, 900)
        self.setMaximumSize(1920, 1080)

        self.config = Config()
        self.scene_key = 'meeting'
        self.freeze = False
        self.show_help = True
        self.running = True
        self.current_page = 1

        self.latest_frame = None
        self.latest_transcript = ''
        self.latest_reply = '等待触发...'
        self.unprocessed_voice_command = None
        self.events: Deque[str] = deque(maxlen=24)
        self.latest_observation = None
        self.fps = 0.0
        self._fps_t0 = time.time()
        self._fps_count = 0
        self._frozen_frame = None
        self.latest_visual_frame = None
        self.gesture_engine = None
        self.engine_crashed = False

        self.state_machine = AppStateMachine()
        self.ai = AIService(self.config)
        self.voice = VoiceService(self.config)
        self.anim_manager = AnimationManager(limit=self.config.max_animations)
        self.pending_voice_command = None
        self.pending_voice_lock = threading.Lock()

        self.latest_gesture = '无'
        self.latest_confidence = 0.0
        self.current_center = (0, 0)
        self.latest_hand_count = 0
        self.last_logged_gesture = None
        self._build_ui()
        self._bind_logic()

        self.cap = self._open_camera()
        if self.cap is None or not self.cap.isOpened():
            self._push_event('摄像头打开失败或被占用', '#EF4444')
            self.cap = None

        self.frame_signal.connect(self._refresh_ui)
        self.log_signal.connect(self._push_event)

        self._push_event('NO-TOUCH 系统已启动', '#7DD3FC')
        self._sync_scene_labels()
        self._apply_state_label('ACTIVE')

        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()
        # ===== 虚拟摄像头初始化（不影响原系统）=====
        try:
            self.virtual_cam = pyvirtualcam.Camera(
                width=self.config.width,
                height=self.config.height,
                fps=30
            )
            self._push_event('虚拟摄像头已启动', '#22C55E')
        except Exception as e:
            self.virtual_cam = None
            self._push_event(f'虚拟摄像头启动失败: {e}', '#EF4444')
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        for btn in self.findChildren(QPushButton):
            btn.setFocusPolicy(Qt.NoFocus)
        for lst in self.findChildren(QListWidget):
            lst.setFocusPolicy(Qt.NoFocus)

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setContentsMargins(18, 18, 18, 18)
        main.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(16)
        left.setContentsMargins(0, 0, 0, 0)

        self.card_scene = GlowCard('#38BDF8', fixed_width=320)
        scene_layout = QVBoxLayout(self.card_scene)
        scene_layout.setContentsMargins(16, 16, 16, 16)
        self.lbl_title = QLabel('无接触多模态助手')
        self.lbl_title.setStyleSheet('font-size: 22px; font-weight: 800; color: #F8FAFC;')
        self.lbl_scene = QLabel('当前场景：')
        self.lbl_scene.setStyleSheet('font-size: 15px; color: #7DD3FC;')
        self.lbl_scene_desc = QLabel('')
        self.lbl_scene_desc.setWordWrap(True)
        self.lbl_scene_desc.setStyleSheet('font-size: 13px; color: #CBD5E1;')
        self.lbl_state = QLabel('● 系统已激活')
        self.lbl_state.setStyleSheet('color: #22C55E; font-weight: 800;')
        scene_layout.addWidget(self.lbl_title)
        scene_layout.addWidget(self.lbl_scene)
        scene_layout.addWidget(self.lbl_scene_desc)
        scene_layout.addWidget(self.lbl_state)
        left.addWidget(self.card_scene)

        self.card_ctrl = GlowCard('#0EA5E9', fixed_width=320)
        ctrl_layout = QVBoxLayout(self.card_ctrl)
        ctrl_layout.setContentsMargins(16, 16, 16, 16)
        ctrl_layout.addWidget(QLabel('快捷控制'))

        row1 = QHBoxLayout()
        for text, key in [('1 会议', 'meeting'), ('2 医疗', 'medical'), ('3 终端', 'kiosk')]:
            btn = QPushButton(text)
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda _, k=key: self._switch_scene(k))
            row1.addWidget(btn)
        ctrl_layout.addLayout(row1)

        row2 = QHBoxLayout()
        for text, cb in [('V 语音', self._start_voice), ('H 帮助', self._toggle_help), ('空格 暂停', self._toggle_pause)]:
            btn = QPushButton(text)
            btn.setFixedHeight(40)
            btn.clicked.connect(cb)
            row2.addWidget(btn)
        ctrl_layout.addLayout(row2)

        row3 = QHBoxLayout()
        for text, cb in [('S 截图', self._save_snapshot), ('R 重置', self._reset_page), ('Q 退出', self.close)]:
            btn = QPushButton(text)
            btn.setFixedHeight(40)
            btn.clicked.connect(cb)
            row3.addWidget(btn)
        ctrl_layout.addLayout(row3)
        left.addWidget(self.card_ctrl)

        self.card_log = GlowCard('#64748B', fixed_width=320)
        log_layout = QVBoxLayout(self.card_log)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(QLabel('最近事件'))
        self.event_log = QListWidget()
        self.event_log.setStyleSheet('background: transparent; color: #E2E8F0; border: none;')
        log_layout.addWidget(self.event_log)
        left.addWidget(self.card_log, 1)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(16)
        center_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        video_card = GlowCard('#38BDF8')
        vc_layout = QVBoxLayout(video_card)
        vc_layout.setContentsMargins(16, 16, 16, 16)
        vc_layout.setSpacing(12)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(900, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            'background: #000000; border-radius: 12px; border: 1px solid rgba(125, 211, 252, 30);'
        )
        vc_layout.addWidget(self.video_label)

        info_strip = GlowCard('#0EA5E9')
        info_strip.setFixedHeight(60)
        istrip_layout = QHBoxLayout(info_strip)
        istrip_layout.setContentsMargins(16, 8, 16, 8)
        self.lbl_gesture = QLabel('手势: 无')
        self.lbl_conf = QLabel('置信度: 0.00')
        self.lbl_hands = QLabel('手数: 0')
        self.lbl_page = QLabel('页码: 1')
        for item in [self.lbl_gesture, self.lbl_conf, self.lbl_hands, self.lbl_page]:
            item.setStyleSheet('font-size: 15px; font-weight: 700;')
            istrip_layout.addWidget(item)
        istrip_layout.addStretch(1)
        vc_layout.addWidget(info_strip)
        center_layout.addWidget(video_card, 1)

        right = QVBoxLayout()
        right.setSpacing(16)
        right.setContentsMargins(0, 0, 0, 0)

        self.card_voice = GlowCard('#38BDF8', fixed_width=360)
        voice_layout = QVBoxLayout(self.card_voice)
        voice_layout.setContentsMargins(16, 16, 16, 16)
        voice_layout.addWidget(QLabel('语音 / AI'))
        self.lbl_voice_status = QLabel('语音: 就绪')
        self.lbl_voice_status.setWordWrap(True)
        self.lbl_voice_status.setStyleSheet('font-size: 14px; color: #F8FAFC;')
        self.lbl_transcript = QLabel('转写: -')
        self.lbl_transcript.setWordWrap(True)
        self.lbl_reply = QLabel('AI 回复: 等待触发...')
        self.lbl_reply.setWordWrap(True)
        self.lbl_reply.setStyleSheet('font-size: 14px; color: #FDE68A;')
        voice_layout.addWidget(self.lbl_voice_status)
        voice_layout.addWidget(self.lbl_transcript)
        voice_layout.addWidget(self.lbl_reply)
        right.addWidget(self.card_voice)

        self.card_help = GlowCard('#64748B', fixed_width=360)
        help_layout = QVBoxLayout(self.card_help)
        help_layout.setContentsMargins(16, 16, 16, 16)
        help_layout.addWidget(QLabel('可用手势'))
        self.help_label = QLabel('')
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet('font-size: 13px; color: #CBD5E1;')
        help_layout.addWidget(self.help_label)
        right.addWidget(self.card_help, 1)

        main.addLayout(left)
        main.addWidget(center_widget, 1)
        main.addLayout(right)

        self.setStyleSheet('''
            QWidget { background-color: #020617; color: #E2E8F0; }
            QPushButton {
                background-color: rgba(15, 23, 42, 210);
                border: 1px solid rgba(125, 211, 252, 80);
                border-radius: 10px;
                color: #E2E8F0;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: rgba(125, 211, 252, 180);
                background-color: rgba(30, 41, 59, 220);
            }
        ''')

    def _bind_logic(self):
        self.state_machine.bind_callback('on_swipe', self._on_swipe)
        self.state_machine.bind_callback('on_confirm', self._on_confirm_trigger)
        self.state_machine.bind_callback('on_multimodal_action', self._on_multimodal_action)
        self.state_machine.bind_callback('on_activate', lambda: self.log_signal.emit('状态机: 激活', '#22C55E'))
        self.state_machine.bind_callback('on_deactivate', lambda: self.log_signal.emit('状态机: 回到空闲', '#94A3B8'))
        self.state_machine.bind_callback('on_state_change', self._on_state_change)

    def _scene(self):
        return SCENES[self.scene_key]

    def _apply_state_label(self, state: str):
        if state == 'ACTIVE':
            text, color = '● 系统已激活', '#22C55E'
        elif state == 'IDLE':
            text, color = '● 系统空闲', '#94A3B8'
        elif state == 'OPERATING':
            text, color = '● 执行中', '#38BDF8'
        elif state == 'COOLDOWN':
            text, color = '● 冷却中', '#F59E0B'
        elif state == 'PAUSED':
            text, color = '● 系统已暂停', '#F59E0B'
        elif state == 'ERROR':
            text, color = '● 识别异常', '#EF4444'
        elif state == '思考中':
            text, color = '● AI 思考中', '#A78BFA'
        else:
            text, color = f'● {state}', '#E2E8F0'
        self.lbl_state.setText(text)
        self.lbl_state.setStyleSheet(f'color: {color}; font-weight: 800;')

    def _push_event(self, text: str, color: str = '#CBD5E1'):
        stamp = time.strftime('%H:%M:%S')
        item = QListWidgetItem(f'[{stamp}] {text}')
        item.setForeground(QColor(color))
        self.event_log.insertItem(0, item)
        while self.event_log.count() > 24:
            self.event_log.takeItem(self.event_log.count() - 1)
        self.events.appendleft(f'[{stamp}] {text}')

    def _sync_scene_labels(self):
        scene = self._scene()
        self.lbl_scene.setText(f'当前场景：{scene["name"]}')
        self.lbl_scene_desc.setText(scene['desc'])
        help_text = '\n'.join([f'{k} → {v}' for k, v in scene['commands'].items()])
        self.help_label.setText(help_text)
        self.lbl_page.setText(f'页码: {self.current_page}')

    def _switch_scene(self, key: str):
        if self.freeze:
            return
        if key not in SCENES:
            return

        self.scene_key = key
        self.current_page = 1
        self._sync_scene_labels()

        self.latest_reply = f'已切换到 {SCENES[key]["name"]}'
        self.lbl_reply.setText(f'AI 回复: {self.latest_reply}')
        self._push_event(f'模式切换: {SCENES[key]["name"]}', '#7DD3FC')

    def _toggle_pause(self):
        self.freeze = not self.freeze
        if self.freeze:
            self.lbl_state.setText('● 系统已暂停')
            self.lbl_state.setStyleSheet('color: #F59E0B; font-weight: 800;')
            self._push_event('画面已暂停')
        else:
            self.lbl_state.setText('● 系统已激活')
            self.lbl_state.setStyleSheet('color: #22C55E; font-weight: 800;')
            self._push_event('画面已恢复')

    def _toggle_help(self):
        self.show_help = not self.show_help
        self._push_event('帮助已切换', '#7DD3FC')

    def _reset_page(self):
        self.current_page = 1
        self.latest_reply = '已重置页面'
        self.lbl_reply.setText(f'AI 回复: {self.latest_reply}')
        self.lbl_page.setText('页码: 1')
        self._push_event('页面已重置', '#7DD3FC')

    def _save_snapshot(self):
        if self.latest_frame is None:
            self._push_event('截图失败: 无视频帧', '#EF4444')
            return
        path = f'snapshot_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
        try:
            cv2.imwrite(path, self.latest_frame)
            self._push_event(f'截图已保存: {path}', '#22C55E')
        except Exception as e:
            self._push_event(f'截图失败: {e}', '#EF4444')

    def _start_voice(self):
        if getattr(self.voice, 'recording', False):
            self._push_event('语音录制中，请稍候...', '#F59E0B')
            return
        if not getattr(self.voice, 'can_record', lambda: False)():
            self._push_event('音频输入不可用或 ASR 未初始化', '#EF4444')
            return
        self._push_event('开始录音', '#F59E0B')
        self.lbl_voice_status.setText('语音: 录音中...')
        self.voice.start_record_and_transcribe_async(
            self.config.voice_record_seconds,
            self.config.sample_rate,
            self._record_done,
        )

    def _record_done(self, transcript: str, err: str):
        if transcript:
            self.latest_transcript = transcript.strip()
            self.latest_reply = '等待处理...'
            with self.pending_voice_lock:
                self.pending_voice_command = self.latest_transcript

            self.log_signal.emit(f'语音输入: {self.latest_transcript}', '#F8FAFC')
            self.lbl_transcript.setText(f'转写: {self.latest_transcript}')
        else:
            self.log_signal.emit(err or '语音识别失败', '#EF4444')
        self._log_event("voice", {
            "user": transcript
        })
        self.lbl_voice_status.setText(f'语音: {getattr(self.voice, "status_text", "就绪")}')
        self.lbl_reply.setText(f'AI 回复: {self.latest_reply}')

    def _on_ai_done(self, reply: str):
        self.latest_reply = reply

        self.log_signal.emit(f'AI 回复: {reply[:42]}', '#F59E0B')

        # 场景化日志
        self._log_event("ai", {
            "user": getattr(self, "latest_transcript", ""),
            "gesture": getattr(self, "latest_frame_gesture", ""),
            "reply": reply,
            "event": f"{self.scene_key.upper()}_AI"
        })

        if getattr(self.config, 'speak_ai_reply', False):
            self.voice.speak_async(reply[:120])

    def _on_state_change(self, state: str):
        self.log_signal.emit(f'状态切换: {state}', '#7DD3FC')

    def _on_confirm_trigger(self):
        self.log_signal.emit('手势已确认', '#22C55E')

    def _on_swipe(self, motion_gesture: str):
        self._apply_command('无', motion_gesture)

    def _on_multimodal_action(self, voice_command: str, static_gesture: str):
        self.log_signal.emit(f'多模态触发: {voice_command}', '#A78BFA')
        self.latest_reply = '思考中...'
        self.ai.ask_async(self.scene_key, static_gesture, voice_command, on_done=self._on_ai_done)

    def trigger_effects(self, gesture: str, center: Tuple[int, int]):
        x, y = center
        px, py = int(x), int(y)

        if gesture == '比心':
            self.anim_manager.add(HeartBurst(px, py))
            self.anim_manager.add(RingPulse(px, py, color=(255, 100, 190), start_radius=16, life=42))
            for _ in range(12):
                self.anim_manager.add(SparkParticle(px, py, color=(255, 140, 200)))

        elif gesture == '点赞':
            self.anim_manager.add(RingPulse(px, py, color=(0, 240, 140), start_radius=12, life=34))
            for _ in range(10):
                self.anim_manager.add(SparkParticle(px, py, color=(0, 240, 140)))

        elif gesture == 'OK':
            self.anim_manager.add(RingPulse(px, py, color=self.config.col_ok))
            for _ in range(10):
                self.anim_manager.add(SparkParticle(px, py, color=(0, 210, 255)))

        elif gesture == '握拳':
            for _ in range(18):
                self.anim_manager.add(SparkParticle(px, py, color=(255, 80, 80)))

        elif gesture in ('五指张开', '双手张开'):
            self.anim_manager.add(RingPulse(px, py, color=self.config.col_accent))

        elif gesture in ('左滑', '右滑'):
            self.anim_manager.add(RingPulse(px, py, color=(255, 220, 80), start_radius=10, life=28))

        elif gesture == '食指指向':
            self.anim_manager.add(RingPulse(px, py, color=(255, 180, 60), start_radius=8, life=26))

    def _apply_command(self, gesture: str, motion_gesture: str = '无'):
        action_gesture = motion_gesture if motion_gesture != '无' else gesture
        if action_gesture == '无':
            return

        changed = False

        if action_gesture in ('向左滑动', '左滑'):
            self.current_page = max(1, self.current_page - 1)
            self.log_signal.emit('执行: 上一页/返回', '#CBD5E1')
            changed = True

        elif action_gesture in ('向右滑动', '右滑'):
            self.current_page += 1
            self.log_signal.emit('执行: 下一页/前进', '#CBD5E1')
            changed = True

        if gesture == '握拳':
            if self.scene_key == 'medical':
                self.latest_reply = '已触发紧急呼叫提醒，请立即通知医护人员。'
                self.log_signal.emit('紧急呼叫已触发', '#EF4444')
                changed = True

        elif gesture == '比心':
            self.log_signal.emit('暖场/安抚反馈已触发', '#F472B6')

        if changed:
            self.lbl_page.setText(f'页码: {self.current_page}')
            self.lbl_reply.setText(f'AI 回复: {self.latest_reply}')

        if action_gesture != self.last_logged_gesture:
            self._log_event("gesture", {
                "event": action_gesture,
                "gesture": gesture
            })
            self.last_logged_gesture = action_gesture

    def _open_camera(self):
        try:
            if platform.system() == 'Windows':
                cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(self.config.camera_index)
            else:
                cap = cv2.VideoCapture(self.config.camera_index)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception:
            return None

    def _draw_empty_screen(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (8, 15, 26)
        frame = put_chinese_text(frame, '无法打开摄像头 或 数据流中断', (70, 120), font_size=30, color=(255, 255, 255))
        frame = put_chinese_text(frame, '请检查摄像头权限、占用情况或 CAMERA_INDEX', (70, 180), font_size=20, color=(220, 220, 220))
        return frame

    def _draw_overlay(self, frame: np.ndarray, gesture: str, score: float, center: Tuple[int, int]):
        h, w = frame.shape[:2]
        frame = put_chinese_text(frame, '无接触多模态交互助手', (24, 18), font_size=24, color=(255, 255, 255))
        frame = put_chinese_text(frame, f'场景: {self._scene()["name"]}', (24, 52), font_size=18, color=(125, 211, 252))
        frame = put_chinese_text(frame, f'识别: {gesture} ({score:.2f})', (w // 2 - 120, 22), font_size=18, color=(120, 240, 120))
        frame = put_chinese_text(frame, f'页码: {self.current_page}', (w - 160, 22), font_size=18, color=(255, 220, 80))
        frame = put_chinese_text(frame, f'语音: {getattr(self.voice, "status_text", "语音: 就绪")}', (24, h - 48), font_size=16, color=(240, 240, 240))
        if self.show_help:
            frame = put_chinese_text(frame, '快捷键：1会议 2医疗 3终端 | V语音 | H帮助 | 空格暂停 | S截图 | R重置 | Q退出', (24, h - 18), font_size=16, color=(255, 255, 255))
        else:
            frame = put_chinese_text(frame, '按 H 显示帮助', (24, h - 18), font_size=16, color=(180, 180, 180))

        cv2.circle(frame, center, 10, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (center[0] - 18, center[1]), (center[0] + 18, center[1]), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (center[0], center[1] - 18), (center[0], center[1] + 18), (0, 220, 255), 1, cv2.LINE_AA)
        return frame

    def _video_loop(self):
        try:
            self.gesture_engine = GestureEngine()
            self.log_signal.emit('手势引擎初始化成功', '#22C55E')
            self.engine_crashed = False
        except Exception as e:
            self.engine_crashed = True
            self.log_signal.emit(f'手势引擎初始化失败: {e}', '#EF4444')
            print(f'引擎初始化报错：{e}')

        
        while self.running:
            if self.cap is None:
                time.sleep(0.05)
                continue

            if self.freeze:
                if self._frozen_frame is None:
                    ok, raw = self.cap.read()
                    if not ok:
                        frame = self._draw_empty_screen()
                    else:
                        frame = cv2.flip(raw, 1)
                        self._frozen_frame = frame.copy()

                frame = self._frozen_frame.copy() if self._frozen_frame is not None else self._draw_empty_screen()
                self.latest_frame = frame.copy()
                self._dispatch_payload(
                    frame,
                    getattr(self, 'latest_frame_gesture', '无'),
                    getattr(self, 'latest_confidence', 0.0),
                    getattr(self, 'current_center', (0, 0)),
                    getattr(self, 'latest_hand_count', 0),
                    'PAUSED',
                )
                if hasattr(self, "virtual_cam"):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.virtual_cam.send(frame_rgb)
                    self.virtual_cam.sleep_until_next_frame()
                time.sleep(0.03)
                continue

            ok, raw = self.cap.read()
            if not ok:
                frame = self._draw_empty_screen()
                self._dispatch_payload(frame, '无', 0.0, (0, 0), 0, 'ERROR')
                time.sleep(0.03)
                continue

            frame = cv2.flip(raw, 1)
            self._frozen_frame = frame.copy()
            self._process_and_dispatch(frame, frozen=False)
            time.sleep(0.001)

    def _process_and_dispatch(self, frame: np.ndarray, frozen: bool):
        # ===== 帧计数器（用于降频）=====
        if not hasattr(self, "frame_id"):
            self.frame_id = 0
        self.frame_id += 1

        gesture, score, center, hands, motion = (
            self.latest_frame_gesture if hasattr(self, "latest_frame_gesture") else '无',
            getattr(self, "latest_confidence", 0.0),
            getattr(self, "current_center", (frame.shape[1] // 2, frame.shape[0] // 2)),
            getattr(self, "latest_hand_count", 0),
            '无'
        )

        if frozen:
            self.latest_frame = frame.copy()
            self._dispatch_payload(
                frame,
                gesture,
                score,
                center,
                hands,
                'PAUSED',
            )
            return

        if getattr(self, 'engine_crashed', False):
            self.latest_frame = frame.copy()
            self._dispatch_payload(frame, '引擎异常', 0.0, center, 0, 'ERROR')
            return

        # ===== 隔帧识别 =====
        do_infer = (self.frame_id % 2 == 0)  # 每2帧识别一次

        if do_infer:
            try:
                if hasattr(self.gesture_engine, 'identify'):
                    obs: FrameObservation = self.gesture_engine.identify(frame)
                    self.latest_observation = obs
                    gesture = obs.primary_gesture
                    score = obs.primary_confidence
                    center = obs.center
                    hands = obs.hand_count
                    motion = obs.motion_gesture

                    # 更新缓存（下一帧用）
                    self.latest_frame_gesture = gesture
                    self.latest_confidence = score
                    self.current_center = center
                    self.latest_hand_count = hands
                    self.latest_visual_frame = frame.copy()
                    if gesture != '无':
                        self.trigger_effects(gesture, center)

                else:
                    gesture, score, center = self.gesture_engine.analyze(frame)
                    self.latest_frame_gesture = gesture
                    self.latest_confidence = score
                    self.current_center = center

            except Exception as e:
                self.engine_crashed = True
                import traceback
                print(traceback.format_exc())
                self.log_signal.emit(f'Engine 推理异常: {str(e)[:60]}', '#EF4444')
                self.latest_frame = frame.copy()
                self._dispatch_payload(frame, '引擎异常', 0.0, center, 0, 'ERROR')
                return
        else:
            # 非推理帧：直接复用上一帧画好线条的图，避免重复调用 draw_landmarks 导致报错
            if self.latest_visual_frame is not None:
                # 使用 np.copyto 快速覆盖当前 frame 的内容，或者直接赋值
                frame[:] = self.latest_visual_frame[:]

                # 同步一下观测数据，保证 UI 文字不闪烁
                if self.latest_observation is not None:
                    obs = self.latest_observation
                    gesture = obs.primary_gesture
                    score = obs.primary_confidence
                    center = obs.center
                    hands = obs.hand_count
            else:
                motion = '无'

        self.latest_frame = frame.copy()

        # ===== 语音事件（一次性消费）=====
        voice_cmd = None
        with self.pending_voice_lock:
            if self.pending_voice_command and not getattr(self.ai, 'is_processing', False):
                voice_cmd = self.pending_voice_command
                self.pending_voice_command = None

        # ===== 状态机 =====
        state = 'ACTIVE'
        try:
            state = self.state_machine.update(
                static_gesture=gesture,
                motion_gesture=motion,
                voice_command=voice_cmd,
                hand_count=hands,
                ai_busy=getattr(self.ai, 'is_processing', False),
            )
        except Exception as e:
            state = 'ERROR'
            self.log_signal.emit(f'状态机错误: {e}', '#EF4444')

        # ===== 执行动作 =====
        if state == 'OPERATING' or motion != '无':
            self._apply_command(gesture, motion)

        # ===== 动画（只调用一次！）=====
        self.anim_manager.update_and_draw(frame)

        # ===== UI绘制 =====
        frame = self._draw_overlay(frame, gesture, score, center)

        # ===== FPS统计 =====
        self._fps_count += 1
        t = time.time()
        if t - self._fps_t0 >= 1.0:
            self.fps = self._fps_count / (t - self._fps_t0)
            self._fps_count = 0
            self._fps_t0 = t

        # ===== 输出 =====
        self._dispatch_payload(frame, gesture, score, center, hands, state)
        # ===== 输出到虚拟摄像头 =====
        if hasattr(self, "virtual_cam") and self.virtual_cam is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 保证尺寸一致
                if frame_rgb.shape[1] != self.virtual_cam.width or frame_rgb.shape[0] != self.virtual_cam.height:
                    frame_rgb = cv2.resize(frame_rgb, (self.virtual_cam.width, self.virtual_cam.height))

                self.virtual_cam.send(frame_rgb)
                self.virtual_cam.sleep_until_next_frame()

            except Exception as e:
                print("虚拟摄像头输出错误:", e)

    def _dispatch_payload(self, frame, gesture, score, center, hands, state):
        self.latest_frame = frame.copy()
        payload = {
            'frame': frame,
            'gesture': gesture,
            'score': score,
            'center': center,
            'hands': hands,
            'state': '思考中' if getattr(self.ai, 'is_processing', False) else state,
            'transcript': self.latest_transcript,
            'reply': self.latest_reply,
            'fps': self.fps,
            'page': self.current_page,
            'voice_status': getattr(self.voice, 'status_text', '语音: 就绪'),
        }
        self.frame_signal.emit(payload)

    def _refresh_ui(self, payload: dict):
        frame = payload.get('frame')
        if frame is None:
            return

        self.lbl_gesture.setText(f"手势: {payload.get('gesture', '无')}")
        self.lbl_conf.setText(f"置信度: {payload.get('score', 0.0):.2f}")
        self.lbl_hands.setText(f"手数: {payload.get('hands', 0)}")
        self.lbl_page.setText(f"页码: {payload.get('page', self.current_page)}")

        self.lbl_voice_status.setText(payload.get('voice_status') or '语音: 就绪')
        self.lbl_transcript.setText(f"转写: {payload.get('transcript') or '-'}")
        self.lbl_reply.setText(f"AI 回复: {payload.get('reply') or '等待触发...'}")

        self._apply_state_label(payload.get('state', 'ACTIVE'))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_1:
            self._switch_scene('meeting')
        elif key == Qt.Key_2:
            self._switch_scene('medical')
        elif key == Qt.Key_3:
            self._switch_scene('kiosk')
        elif key == Qt.Key_V:
            self._start_voice()
        elif key == Qt.Key_H:
            self._toggle_help()
        elif key == Qt.Key_S:
            self._save_snapshot()
        elif key == Qt.Key_R:
            self._reset_page()
        elif key == Qt.Key_Space:
            self._toggle_pause()
        elif key in (Qt.Key_Escape, Qt.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            handled = True

            if key == Qt.Key_1:
                self._switch_scene('meeting')
            elif key == Qt.Key_2:
                self._switch_scene('medical')
            elif key == Qt.Key_3:
                self._switch_scene('kiosk')
            elif key == Qt.Key_V:
                self._start_voice()
            elif key == Qt.Key_H:
                self._toggle_help()
            elif key == Qt.Key_S:
                self._save_snapshot()
            elif key == Qt.Key_R:
                self._reset_page()
            elif key == Qt.Key_Space:
                self._toggle_pause()
            elif key in (Qt.Key_Escape, Qt.Key_Q):
                self.close()
            else:
                handled = False

            if handled:
                return True

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        event.accept()

    def _log_event(self, event_type: str, content: dict):
        import os
        from datetime import datetime

        os.makedirs("logs", exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"logs/{self.scene_key}_{date_str}.txt"

        time_str = datetime.now().strftime("%H:%M:%S")

        # ===== 不同场景格式化 =====
        log_text = ""

        if self.scene_key == "meeting":
            log_text = self._format_meeting_log(time_str, event_type, content)

        elif self.scene_key == "medical":
            log_text = self._format_medical_log(time_str, event_type, content)

        else:  # kiosk / 默认
            log_text = self._format_general_log(time_str, event_type, content)

        with open(filename, "a", encoding="utf-8") as f:
            f.write(log_text)

    def _format_meeting_log(self, time_str, event_type, content):
        if event_type == "voice":
            return f"[{time_str}] 用户：{content.get('user', '')}\n"

        elif event_type == "gesture":
            return f"[{time_str}] 手势：{content.get('gesture', '')}\n"

        elif event_type == "ai":
            return (
                f"[{time_str}]\n"
                f"AI总结：\n{content.get('reply', '')}\n"
                "-----------------------------------\n"
            )

        return ""

    def _format_medical_log(self, time_str, event_type, content):
        return (
            f"[{time_str}]\n"
            f"事件：{content.get('event', '')}\n"
            f"手势：{content.get('gesture', '')}\n"
            f"系统：{content.get('reply', '')}\n"
            "-----------------------------------\n"
        )

    def _format_general_log(self, time_str, event_type, content):
        return (
            f"[{time_str}] "
            f"{content.get('event', '')} | 手势:{content.get('gesture', '')}\n"
        )

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass

    app = QApplication(sys.argv)
    font = QFont('Microsoft YaHei', 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)

    try:
        import _locale
        _locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
    except Exception:
        pass

    win = TouchlessAssistantApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()