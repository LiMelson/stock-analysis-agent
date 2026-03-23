"""
Web Search 工具模块

使用 Tavily API 进行实时网络搜索。
支持自动添加时间上下文，提高时效性搜索的准确性。
"""

import os
import re
from datetime import datetime
from typing import Optional, List
from tavily import TavilyClient


def get_current_date() -> str:
    """获取当前日期，格式：2026年03月22日"""
    return datetime.now().strftime("%Y年%m月%d日")


def add_time_context(query: str) -> str:
    """
    为搜索查询添加时间上下文
    
    如果查询包含时效性关键词，自动添加当前日期
    """
    # 时效性关键词
    time_keywords = [
        "今天", "今日", "现在", "当前", "最新", "最近", "近日", "近期",
        "今天行情", "今日股价", "最新消息", "最新动态", "最新行情",
        "今天走势", "今日走势", "现在价格", "当前价格"
    ]
    
    # 检查是否包含时效性关键词
    has_time_keyword = any(keyword in query for keyword in time_keywords)
    
    # 如果包含时效性关键词，添加当前日期
    if has_time_keyword:
        current_date = get_current_date()
        # 避免重复添加日期
        if current_date[:4] not in query:  # 简单检查是否已包含年份
            return f"{current_date} {query}"
    
    return query


class WebSearchTool:
    """基于 Tavily 的实时网络搜索工具（支持时间上下文）"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 WebSearchTool
        
        Args:
            api_key: Tavily API Key，如果不提供则从环境变量 TAVILY_API_KEY 获取
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API Key 未提供，请设置 TAVILY_API_KEY 环境变量或直接传入 api_key")
        
        self.client = TavilyClient(api_key=self.api_key)
    
    def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 5,
        include_answer: str = "advanced",
        include_raw_content: bool = False,
        add_time: bool = True
    ) -> dict:
        """
        执行实时网络搜索
        
        Args:
            query: 搜索查询内容
            search_depth: 搜索深度
            max_results: 返回的最大结果数量，默认为 5
            include_answer: 包含 AI 生成的答案摘要
            include_raw_content: 是否包含原始网页内容，默认为 False
            add_time: 是否自动添加时间上下文，默认为 True
            
        Returns:
            dict: Tavily API 返回的搜索结果
        """
        # 添加时间上下文
        if add_time:
            query = add_time_context(query)
        
        try:
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer,
                include_raw_content=include_raw_content
            )
            return response
        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def search_contents(
        self,
        query: str,
        max_results: int = 5,
        add_time: bool = True
    ) -> List[str]:
        """
        执行搜索并返回 content 列表
        
        Args:
            query: 搜索查询内容
            max_results: 返回的最大结果数量
            add_time: 是否自动添加时间上下文，默认为 True
            
        Returns:
            List[str]: 每个搜索结果的 content 列表
        """
        response = self.search(
            query=query,
            add_time=add_time,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False
        )
        
        if "error" in response:
            return []
        
        contents = []
        for result in response.get("results", []):
            content = result.get("content", "")
            if content:
                contents.append(content)
        
        return contents
