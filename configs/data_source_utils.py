"""
数据源工具模块

提供数据源通用的工具函数和装饰器
"""

import time
import functools
from typing import Callable, Any


def retry_on_network_error(max_retries: int = 3, delay: float = 1.0):
    """
    网络错误重试装饰器
    
    用于处理 tushare 等数据源的临时网络错误
    
    Args:
        max_retries: 最大重试次数
        delay: 初始重试间隔（秒），会指数退避
        
    使用示例:
        @retry_on_network_error(max_retries=3, delay=1.0)
        def fetch_data():
            return ak.index_zh_a_hist(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # 判断是否是网络相关错误（可重试）
                    retryable_errors = [
                        'timeout', 'connection', 'network', 'temporarily',
                        'rate limit', '503', '502', '504', '503',
                        'refused', 'reset', 'broken pipe'
                    ]
                    is_retryable = any(err in error_msg for err in retryable_errors)
                    
                    if not is_retryable:
                        # 不可重试的错误（如股票代码不存在），直接抛出
                        raise
                    
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # 指数退避
                        print(f"  [网络超时] {wait_time:.1f}秒后重试 ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        print(f"  [网络错误] 重试{max_retries}次后仍失败")
            
            # 所有重试都失败，返回友好的错误信息
            raise Exception(f"网络连接失败，请检查网络后重试。错误: {last_exception}")
        
        return wrapper
    return decorator


def format_data_error(source_name: str, error: Exception) -> dict:
    """
    格式化数据源错误为统一格式
    
    Args:
        source_name: 数据源名称
        error: 异常对象
        
    Returns:
        统一格式的错误响应
    """
    error_msg = str(error).lower()
    
    # 判断错误类型
    if any(x in error_msg for x in ['timeout', 'connection', 'network', 'refused']):
        user_msg = f"📡 [{source_name}] 网络连接失败，请检查网络后重试"
    elif 'no module' in error_msg or 'import' in error_msg:
        user_msg = f"📦 [{source_name}] 缺少依赖包，请执行: pip install tushare pandas"
    elif 'code' in error_msg or 'symbol' in error_msg or '股票' in error_msg:
        user_msg = f"📋 [{source_name}] 股票代码错误或数据不存在"
    else:
        user_msg = f"⚠️ [{source_name}] 数据获取失败: {error}"
    
    return {
        "status": "error",
        "error": str(error),
        "message": user_msg,
        "data": None
    }


import concurrent.futures

def safe_fetch(fetch_func: Callable, source_name: str, *args, timeout: int = 30, **kwargs) -> dict:
    """
    安全执行数据获取，带超时控制
    
    Args:
        fetch_func: 数据获取函数
        source_name: 数据源名称
        timeout: 超时时间（秒），默认30秒
        *args, **kwargs: 传给 fetch_func 的参数
        
    Returns:
        数据结果或错误信息
    """
    print(f"  [INFO] 正在获取 {source_name} 数据...", flush=True)
    
    try:
        # 使用线程池实现超时控制
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fetch_func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                if isinstance(result, dict) and "status" in result:
                    print(f"  [OK] {source_name} 数据获取成功")
                    return result
                print(f"  [OK] {source_name} 数据获取成功")
                return {"status": "success", "data": result}
            except concurrent.futures.TimeoutError:
                print(f"  [TIMEOUT] {source_name} 获取超时({timeout}秒)")
                return {
                    "status": "error", 
                    "error": f"{source_name} 获取超时",
                    "message": f"[{source_name}] 数据获取超时，请检查网络连接",
                    "data": None
                }
    except Exception as e:
        print(f"  [ERROR] {source_name} 获取失败: {e}")
        return format_data_error(source_name, e)
