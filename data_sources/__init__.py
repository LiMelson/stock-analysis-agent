"""
Data Sources 模块

提供股票分析所需的数据源：
- rag: 知识库检索（个股基本面、投资理论）
- index: 指数数据（上证指数、深证成指、创业板指等）
- sentiment: 市场情绪数据（同花顺情绪指数）
- theme: 题材/板块数据（热点板块、资金流向）
- stock: 个股数据（实时行情、历史K线、财务数据）
"""

from .rag import RAGDataSource
from .index import IndexDataSource
from .sentiment import SentimentDataSource
from .theme import ThemeDataSource
from .stock import StockDataSource

__all__ = [
    "RAGDataSource",
    "IndexDataSource", 
    "SentimentDataSource",
    "ThemeDataSource",
    "StockDataSource"
]
