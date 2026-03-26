"""
模型配置模块
LLM的初始化以及Embedding的初始化
提供 Embedding 模型和 LLM 客户端的统一管理。
"""

import os

# 设置 Hugging Face 镜像源（必须在导入 sentence_transformers 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置模型缓存目录为项目本地目录（避免中文路径问题）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(project_root, 'model_cache')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import warnings
from typing import List, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 尝试导入可选依赖
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    model_name: str = 'BAAI/bge-small-zh-v1.5'
    device: str = 'cpu'
    normalize_embeddings: bool = True
    batch_size: int = 32


class EmbeddingModel:
    """
    Embedding 模型管理器（项目最终精简版）
    仅支持：Sentence Transformers 模型
    """

    def __init__(self, config=None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.dimension = 512

        # 必须安装依赖
        if not HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("[FAIL] 请安装：pip install sentence-transformers")

        # 必须加载成功，失败直接报错
        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"[OK] 已加载 Embedding 模型: {self.config.model_name} (维度: {self.dimension})")
        except Exception as e:
            raise RuntimeError(f"[FAIL] 模型加载失败: {e}") from e

    def encode(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dimension
        emb = self.model.encode(text, normalize_embeddings=self.config.normalize_embeddings)
        return emb.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embs = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=len(texts) > 100
        )
        return embs.tolist()

    def get_dimension(self) -> int:
        return self.dimension
    
    def get_sentence_embedding_dimension(self) -> int:
        """兼容方法，与 SentenceTransformer 接口一致"""
        return self.dimension


class LLMClient:
    """
    LLM 客户端封装
    
    用于调用大语言模型生成文本。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: API 密钥，默认从环境变量 API_KEY 读取
            base_url: API 基础 URL，默认从环境变量 BASE_URL 读取
            model: 模型名称，默认从环境变量 MODEL 读取
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")
        self.model = model or os.getenv("MODEL")
        
        if not self.api_key:
            raise ValueError("请提供 api_key 或设置 API_KEY 环境变量")
        
        self.client = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60,  # 60秒超时
            max_retries=2  # 最多重试2次
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        stream_callback=None
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度参数（默认 0.3，适合分析任务）
                - 0.0-0.3: 高确定性，适合分析/报告（默认）
                - 0.3-0.5: 平衡，适合总结
                - 0.5-0.7: 一定创造性，适合查询扩展
            max_tokens: 最大生成token数
            stream: 是否使用流式输出
            stream_callback: 流式输出回调函数，接收每个 token
            
        Returns:
            生成的文本
        """
        # 构建 LangChain 消息格式
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        if stream and stream_callback:
            # 流式输出
            full_response = ""
            for chunk in self.client.stream(
                messages,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                content = chunk.content
                if content:
                    full_response += content
                    stream_callback(content)
            return full_response.strip()
        else:
            # 非流式输出
            response = self.client.invoke(
                messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content.strip()
    
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ):
        """
        流式生成文本，返回生成器
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度参数（默认 0.3，适合分析任务）
            max_tokens: 最大生成token数
            
        Yields:
            每个生成的 token
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        for chunk in self.client.stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            if chunk.content:
                yield chunk.content