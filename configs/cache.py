"""
简单缓存模块

用于缓存 tushare 等数据源的结果，避免频繁请求
"""

import time
from typing import Dict, Any, Optional
from functools import wraps


class DataCache:
    """
    简单内存缓存
    
    缓存时间：5分钟（300秒）
    """
    
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key not in self.cache:
            return None
        
        item = self.cache[key]
        if time.time() - item["timestamp"] > item["ttl"]:
            # 过期了
            del self.cache[key]
            return None
        
        return item["data"]
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None):
        """设置缓存数据"""
        self.cache[key] = {
            "data": data,
            "timestamp": time.time(),
            "ttl": ttl or self.default_ttl
        }
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()


# 全局缓存实例
_cache = DataCache()


def cached(ttl: int = 300, key_prefix: str = ""):
    """
    缓存装饰器
    
    Args:
        ttl: 缓存时间（秒），默认5分钟
        key_prefix: 缓存键前缀
        
    使用示例:
        @cached(ttl=300, key_prefix="index")
        def fetch_index_data():
            return ak.index_zh_a_hist(...)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 尝试从缓存获取
            cached_data = _cache.get(cache_key)
            if cached_data is not None:
                print(f"  [CACHE] 使用缓存数据: {key_prefix}")
                return cached_data
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            _cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def get_cache() -> DataCache:
    """获取全局缓存实例"""
    return _cache
