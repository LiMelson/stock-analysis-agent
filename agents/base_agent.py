"""
Base Agent 模块

提供所有 Agent 的基类，包含：
- 日志记录
- 异常处理
- 耗时统计
"""

import time
import logging
import traceback
from typing import Any, Dict, Optional, Callable
from functools import wraps


# 配置日志 - 默认只输出 WARNING 及以上级别，隐藏 INFO 调试信息
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)


class AgentLogger:
    """Agent 专用日志记录器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)


class AgentTimer:
    """耗时统计工具"""
    
    def __init__(self, name: str, logger: AgentLogger):
        self.name = name
        self.logger = logger
        self.start_time: Optional[float] = None
        self.steps: Dict[str, float] = {}
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_time = time.time()
        self.logger.info(f"[{self.name}] 开始执行")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.logger.info(f"[{self.name}] 执行完成，总耗时: {total_time:.3f}s")
            
            # 输出各步骤耗时
            if self.steps:
                self.logger.info(f"[{self.name}] 各步骤耗时:")
                for step_name, step_time in self.steps.items():
                    self.logger.info(f"  - {step_name}: {step_time:.3f}s")
    
    def step(self, step_name: str):
        """记录某一步骤的耗时"""
        return _StepTimer(self, step_name)


class _StepTimer:
    """单步骤耗时计时器"""
    
    def __init__(self, agent_timer: AgentTimer, step_name: str):
        self.agent_timer = agent_timer
        self.step_name = step_name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.agent_timer.steps[self.step_name] = elapsed


def agent_run_wrapper(func: Callable) -> Callable:
    """
    Agent run 方法装饰器
    
    提供统一的：
    - 异常捕获
    - 耗时统计
    - 日志记录
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        agent_name = self.__class__.__name__
        
        # 记录输入参数（隐藏敏感信息）
        safe_kwargs = {k: v for k, v in kwargs.items() if 'key' not in k.lower()}
        self.logger.info(f"[{agent_name}] 调用参数: {safe_kwargs}")
        
        try:
            with AgentTimer(agent_name, self.logger):
                result = func(self, *args, **kwargs)
                
            # 记录结果摘要
            if isinstance(result, dict):
                result_keys = list(result.keys())
                self.logger.info(f"[{agent_name}] 返回结果字段: {result_keys}")
            elif isinstance(result, str):
                preview = result[:100] + "..." if len(result) > 100 else result
                self.logger.info(f"[{agent_name}] 返回结果: {preview}")
            else:
                self.logger.info(f"[{agent_name}] 返回类型: {type(result).__name__}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{agent_name}] 执行异常: {type(e).__name__}: {str(e)}")
            self.logger.error(f"[{agent_name}] 异常堆栈:\n{traceback.format_exc()}")
            
            # 返回友好的错误信息
            return {
                "error": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "answer": f"处理失败: {str(e)}"
            }
    
    return wrapper


class BaseAgent:
    """
    所有 Agent 的基类
    
    提供日志、异常处理、耗时统计功能
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        初始化 Base Agent
        
        Args:
            name: Agent 名称，默认使用类名
        """
        self.name = name or self.__class__.__name__
        self.logger = AgentLogger(self.name)
        self.logger.info(f"[{self.name}] Agent 初始化完成")
    
    def _safe_run(self, func: Callable, *args, **kwargs) -> Any:
        """
        安全执行函数，自动处理异常和日志
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数
            
        Returns:
            函数返回值，或错误信息字典
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"[{self.name}] 调用失败: {type(e).__name__}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _log_step(self, step_name: str, message: str = ""):
        """记录执行步骤"""
        if message:
            self.logger.info(f"[{self.name}] {step_name}: {message}")
        else:
            self.logger.info(f"[{self.name}] {step_name}")
