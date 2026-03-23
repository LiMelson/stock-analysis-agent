"""
配置模块

提供模型配置。
"""

from configs.model_config import EmbeddingModel, LLMClient, EmbeddingConfig

__all__ = [
    # Embedding 配置
    "EmbeddingConfig",
    # 模型
    "EmbeddingModel",
    # LLM
    "LLMClient",
]
