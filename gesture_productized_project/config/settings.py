from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


@dataclass
class Config:
    # 摄像头 / 画面
    camera_index: int = _env_int('CAMERA_INDEX', 0)
    fps: int = _env_int('FPS', 30)
    width: int = _env_int('WIDTH', 1280)
    height: int = _env_int('HEIGHT', 720)

    # 语音
    voice_record_seconds: float = _env_float('VOICE_RECORD_SECONDS', 4.0)
    sample_rate: int = _env_int('SAMPLE_RATE', 16000)

    # LLM
    llm_api_key: str = os.getenv('LLM_API_KEY', 'sk-***********0b6')#请在此处填入你的api_key
    llm_base_url: str = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
    llm_model: str = os.getenv('LLM_MODEL', 'deepseek-chat')

    # ASR
    asr_api_key: str = os.getenv('ASR_API_KEY', 'sk-cc***********eb233')#请在此处填入你的api_key
    asr_base_url: str = os.getenv('ASR_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    asr_model: str = os.getenv('ASR_MODEL', 'qwen3-asr-flash')

    # 交互
    tts_enabled: bool = os.getenv('TTS_ENABLED', '1') != '0'
    speak_ai_reply: bool = os.getenv('SPEAK_AI_REPLY', '1') != '0'
    scene_default: str = os.getenv('SCENE_DEFAULT', 'meeting')

    # 手势识别
    gesture_smooth_window: int = _env_int('GESTURE_SMOOTH_WINDOW', 7)
    gesture_cooldown: float = _env_float('GESTURE_COOLDOWN', 0.55)
    trail_length: int = _env_int('TRAIL_LENGTH', 14)
    max_animations: int = _env_int('MAX_ANIMATIONS', 50)

    # 颜色（BGR）
    col_text: tuple[int, int, int] = (255, 255, 255)
    col_bg: tuple[int, int, int] = (26, 30, 39)
    col_accent: tuple[int, int, int] = (0, 220, 255)
    col_ok: tuple[int, int, int] = (80, 220, 120)
    col_warn: tuple[int, int, int] = (0, 180, 255)
    col_alert: tuple[int, int, int] = (60, 60, 255)


SCENES: Dict[str, dict] = {
    'meeting': {
        'name': '会议协作模式',
        'desc': '无接触翻页、提问总结、确认与暂停',
        'prompt': (
            '你是一个会议场景下的多模态助手。用户正在进行演示、汇报或线上会议，'
            '请用简短、明确、自然的中文回复。优先输出适合会议的建议、总结和下一步行动。'
        ),
        'commands': {
            '左滑': '上一页 / 返回',
            '右滑': '下一页 / 前进',
            'OK': '确认 / 继续',
            '点赞': '确认理解 / 赞同',
            '食指指向': '激光指示 / 聚焦',
            '五指张开': '呼出菜单 / 全屏提示',
            '比心': '生成暖场反馈',
        },
    },
    'medical': {
        'name': '医疗陪护模式',
        'desc': '无接触呼叫、状态确认、症状描述',
        'prompt': (
            '你是一个医疗陪护与病房辅助助手。回复要非常简洁、平稳、清晰，优先给出确认、提醒和安全建议。'
            '如果出现急症相关内容，要建议立即呼叫医护人员。'
        ),
        'commands': {
            '握拳': '紧急呼叫',
            'OK': '确认已收到',
            '点赞': '状态正常',
            '食指指向': '查看详情',
            '五指张开': '展开提醒',
            '比心': '安抚反馈',
        },
    },
    'kiosk': {
        'name': '公共终端模式',
        'desc': '无接触菜单导航、信息查询、确认提交',
        'prompt': (
            '你是一个公共服务终端助手。请用短句帮助用户完成菜单导航、信息查询和确认操作。'
            '回复要清晰、礼貌、可执行。'
        ),
        'commands': {
            '左滑': '上一个菜单',
            '右滑': '下一个菜单',
            'OK': '确认选择',
            '点赞': '满意 / 确认',
            '握拳': '取消 / 返回',
            '五指张开': '主页 / 菜单总览',
        },
    },
}
