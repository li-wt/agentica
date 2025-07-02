# -*- coding: utf-8 -*-
"""
应用配置文件
用于管理环境变量和应用设置
"""
import os
from typing import Dict, Any

class AppConfig:
    """应用配置类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 模型配置
        "default_model": "gpt-4o-mini",
        "available_models": {
            "gpt-4o-mini": "GPT-4O Mini (推荐)",
            "gpt-4o": "GPT-4O (强大)", 
            "gpt-3.5-turbo": "GPT-3.5 Turbo (经济)"
        },
        
        # 工具配置
        "default_tools": ["calculator", "weather", "random", "text_processor"],
        "available_tools": {
            'calculator': '🧮 计算器',
            'weather': '🌤️ 天气查询',
            'search': '🔍 网络搜索',
            'file': '📁 文件操作',
            'random': '🎲 随机工具',
            'text_processor': '📝 文本处理',
            'shell': '🖥️ 系统命令',
            'code': '💻 代码执行'
        },
        
        # 存储配置
        "storage": {
            "db_file": "outputs/chat_app.db",
            "lance_db_uri": "outputs/chat_app_lancedb",
            "table_name": "chat_documents"
        },
        
        # UI配置
        "ui": {
            "page_title": "Agentica 智能助手",
            "page_icon": "🤖",
            "layout": "wide",
            "show_intent_by_default": True,
            "debug_mode_by_default": False
        }
    }
    
    @classmethod
    def get_openai_api_key(cls) -> str:
        """获取 OpenAI API Key"""
        return os.getenv("OPENAI_API_KEY", "")
    
    @classmethod
    def get_serper_api_key(cls) -> str:
        """获取 Serper API Key (用于网络搜索)"""
        return os.getenv("SERPER_API_KEY", "")
    
    @classmethod
    def get_weather_api_key(cls) -> str:
        """获取天气 API Key"""
        return os.getenv("WEATHER_API_KEY", "")
    
    @classmethod
    def has_required_api_keys(cls) -> Dict[str, bool]:
        """检查必要的 API Keys 是否配置"""
        return {
            "openai": bool(cls.get_openai_api_key()),
            "serper": bool(cls.get_serper_api_key()),
            "weather": bool(cls.get_weather_api_key())
        }
    
    @classmethod
    def get_config(cls, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        config = cls.DEFAULT_CONFIG
        
        for k in keys:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                return default
        
        return config

    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("logs", exist_ok=True) 