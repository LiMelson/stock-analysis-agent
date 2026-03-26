"""
Data Source Router 模块

职责：根据意图类型选择数据源
- 输入：intent_type（来自 IntentRequirementAgent）
- 输出：required_sources + queries

规则驱动，无需LLM
"""

from typing import Dict, Any, Optional, List


class DataSourceRouter:
    """
    数据源路由器
    
    根据意图类型映射到数据源，纯规则驱动
    """
    
    # 意图类型到数据源的映射规则
    INTENT_SOURCE_MAP = {
        "real_time_quote": {
            "sources": ["stock", "index"],
            "description": "实时行情需要个股数据+大盘指数"
        },
        "historical_report": {
            "sources": ["rag", "stock"],
            "description": "历史数据需要知识库（基本面）+个股数据"
        },
        "investment_analysis": {
            "sources": ["rag", "index", "sentiment", "theme", "stock"],
            "description": "投资分析需要知识库+全量市场数据综合分析"
        },
        "general_question": {
            "sources": ["rag"],
            "description": "一般问题通过知识库回答（投资理论、概念解释）"
        },
        "greeting": {
            "sources": [],
            "description": "问候不需要外部数据"
        }
    }
    
    # 查询模板
    QUERY_TEMPLATES = {
        "rag": "{target} {topic} 基本面 投资知识",
        "index": "{date} {target}大盘走势 指数分析",
        "sentiment": "{date} A股市场情绪 涨跌家数",
        "theme": "{date} 热点板块 题材概念",
        "stock": "{date} {target}股票走势 行情"
    }
    
    def route(
        self,
        question: str,
        intent_type: str,
        requirement_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        路由决策 - 根据意图类型选择数据源
        
        Args:
            question: 用户问题
            intent_type: 意图类型（来自 IntentRequirementAgent）
            requirement_analysis: 需求分析结果
            
        Returns:
            包含数据源选择和查询语句的字典
        """
        # 获取目标股票和关注点
        target_stock = requirement_analysis.get("target_stock", "")
        focus_areas = requirement_analysis.get("focus_areas", [])
        target = target_stock if target_stock else ""
        topic = focus_areas[0] if focus_areas else ""
        
        # 根据意图类型获取数据源
        mapping = self.INTENT_SOURCE_MAP.get(intent_type, {"sources": [], "description": "未知意图，不选择数据源"})
        required_sources = mapping["sources"]
        
        # 生成查询语句
        from datetime import datetime
        date = datetime.now().strftime("%Y年%m月%d日")
        
        queries = {}
        for source in required_sources:
            template = self.QUERY_TEMPLATES.get(source, "")
            if template:
                queries[f"{source}_query"] = template.format(date=date, target=target, topic=topic)
        
        # 填充未选中的查询为空字符串
        for source in ["rag", "index", "sentiment", "theme", "stock"]:
            key = f"{source}_query"
            if key not in queries:
                queries[key] = ""
        
        return {
            "required_sources": required_sources,
            "rag_query": queries.get("rag_query", ""),
            "index_query": queries.get("index_query", ""),
            "sentiment_query": queries.get("sentiment_query", ""),
            "theme_query": queries.get("theme_query", ""),
            "stock_query": queries.get("stock_query", "")
        }
