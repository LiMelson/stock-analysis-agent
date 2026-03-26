"""
Data Source Router 模块 - 数据源路由 Agent

职责：
- 根据意图类型和需求选择数据源
- 生成针对各数据源的查询语句
- 为后续接入新数据源（如金融API、公告数据库）预留扩展点

当前支持的数据源：
- rag: 知识库检索
- search: 实时搜索（Tavily）

预留扩展数据源：
- financial_api: 金融数据API
- announcement_db: 公告数据库
- market_data: 市场行情数据
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from configs.model_config import LLMClient
from agents.base_agent import BaseAgent, agent_run_wrapper, AgentTimer


class DataSourceRouter(BaseAgent):
    """
    数据源路由 Agent - 决定需要什么数据源
    
    作为 PlanAgent 的子模块，负责：
    - 根据意图类型和需求选择数据源
    - 生成针对各数据源的查询语句
    - 为后续接入新数据源（如金融API、公告数据库）预留扩展点
    """
    
    # 数据源类型定义（便于后续扩展）
    SOURCE_RAG = "rag"
    SOURCE_SEARCH = "search"
    SOURCE_FINANCIAL_API = "financial_api"      # 预留：金融API
    SOURCE_ANNOUNCEMENT_DB = "announcement_db"  # 预留：公告数据库
    SOURCE_MARKET_DATA = "market_data"          # 预留：市场行情
    
    def __init__(self):
        super().__init__("DataSourceRouter")
        self.llm_client = LLMClient()
    
    @agent_run_wrapper
    def route(
        self,
        question: str,
        intent_type: str,
        requirement_analysis: Dict[str, Any],
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        路由决策 - 决定需要什么数据源
        
        Args:
            question: 用户问题
            intent_type: 意图类型
            requirement_analysis: 需求分析结果
            chat_history: 历史对话记录
            
        Returns:
            包含路由决策结果的字典
        """
        self._log_step("数据源路由", f"intent={intent_type}")
        
        # 根据意图类型预设数据源优先级
        source_priority = self._get_source_priority_by_intent(intent_type)
        
        system_prompt = f"""你是一位数据源路由专家。根据用户意图和需求，决定需要什么数据源。

## 可用数据源

### 当前已实现
1. **rag** - 知识库检索
   - 适用场景：历史财报、经典投资理论、基本面分析知识
   - 数据内容：已收录的财报数据、投资理论、公司分析

2. **search** - 实时搜索（Tavily）
   - 适用场景：最新股价、新闻动态、市场情绪、政策变化
   - 数据内容：互联网实时信息

### 预留扩展（规划中）
3. **financial_api** - 金融数据API
   - 适用场景：结构化财务数据、实时行情、历史K线
   - 数据内容：财务报表、估值指标、行情数据

4. **announcement_db** - 公告数据库
   - 适用场景：公司公告、重大事项、披露信息
   - 数据内容：上市公司公告、监管信息

5. **market_data** - 市场行情数据
   - 适用场景：实时行情、板块数据、资金流向
   - 数据内容：实时股价、成交量、资金流向

## 意图类型对应的数据源优先级
{source_priority}

当前时间：{datetime.now().strftime('%Y年%m月')}"""

        prompt = f"""请根据以下信息，决定需要什么数据源并生成查询语句。

用户问题：{question}
意图类型：{intent_type}
需求分析：{requirement_analysis}

请输出：

```
REQUIRED_SOURCES: ["rag"] 或 ["search"] 或 ["rag", "search"] 或 []
RAG_QUERY: <知识库查询语句>
SEARCH_QUERY: <搜索查询语句>
FINANCIAL_API_QUERY: <金融API查询语句（如有需要）>
ANNOUNCEMENT_QUERY: <公告查询语句（如有需要）>
```

示例1（仅需知识库）：
```
REQUIRED_SOURCES: ["rag"]
RAG_QUERY: 贵州茅台2023年财务报表 净利润 营收
SEARCH_QUERY: 
FINANCIAL_API_QUERY: 
ANNOUNCEMENT_QUERY: 
```

示例2（仅需搜索）：
```
REQUIRED_SOURCES: ["search"]
RAG_QUERY: 
SEARCH_QUERY: 贵州茅台股票今日行情 股价走势 最新消息
FINANCIAL_API_QUERY: 
ANNOUNCEMENT_QUERY: 
```

示例3（两者都需要）：
```
REQUIRED_SOURCES: ["rag", "search"]
RAG_QUERY: 贵州茅台 财报数据 盈利能力 ROE 毛利率
SEARCH_QUERY: 贵州茅台今日股价 新闻动态 市场表现
FINANCIAL_API_QUERY: 
ANNOUNCEMENT_QUERY: 
```"""

        with AgentTimer("路由决策", self.logger):
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
        
        # 解析结果
        parsed = self._parse_routing_output(response)
        
        self.logger.info(f"路由决策: {parsed['required_sources']}")
        
        return parsed
    
    def _get_source_priority_by_intent(self, intent_type: str) -> str:
        """根据意图类型获取数据源优先级说明"""
        priorities = {
            "real_time_quote": """
- 首选：search（获取最新行情）
- 次选：market_data（预留，获取实时数据）
- 备选：financial_api（预留，获取历史对比）""",
            
            "historical_report": """
- 首选：rag（历史财报数据）
- 次选：financial_api（预留，结构化财务数据）
- 备选：announcement_db（预留，相关公告）""",
            
            "investment_analysis": """
- 默认：rag + search（综合分析）
- 基本面侧重：rag + financial_api
- 消息面侧重：search + announcement_db""",
            
            "general_question": """
- 首选：rag（投资理论知识）
- 一般不需要其他数据源""",
            
            "greeting": """
- 不需要任何数据源
- 直接由LLM回复"""
        }
        
        return priorities.get(intent_type, "- 默认：rag + search")
    
    def _parse_routing_output(self, text: str) -> Dict[str, Any]:
        """解析路由输出"""
        import re
        
        result = {
            "required_sources": [],
            "rag_query": "",
            "search_query": "",
            "financial_api_query": "",
            "announcement_query": "",
            "market_data_query": ""
        }
        
        # 解析 REQUIRED_SOURCES
        sources_match = re.search(r'REQUIRED_SOURCES:\s*(\[[^\]]*\])', text, re.IGNORECASE)
        if sources_match:
            sources_str = sources_match.group(1)
            if '"rag"' in sources_str or "'rag'" in sources_str:
                result["required_sources"].append("rag")
            if '"search"' in sources_str or "'search'" in sources_str:
                result["required_sources"].append("search")
            if '"financial_api"' in sources_str or "'financial_api'" in sources_str:
                result["required_sources"].append("financial_api")
            if '"announcement_db"' in sources_str or "'announcement_db'" in sources_str:
                result["required_sources"].append("announcement_db")
            if '"market_data"' in sources_str or "'market_data'" in sources_str:
                result["required_sources"].append("market_data")
        
        # 如果没匹配到，根据关键词判断
        if not result["required_sources"]:
            text_lower = text.lower()
            realtime_keywords = ["最新", "今天", "现在", "当前", "行情", "实时", "新闻", "动态"]
            need_realtime = any(kw in text_lower for kw in realtime_keywords)
            history_keywords = ["历史", "基本面", "财报", "理论", "估值", "分析"]
            need_history = any(kw in text_lower for kw in history_keywords)
            
            if need_realtime:
                result["required_sources"].append("search")
            if need_history:
                result["required_sources"].append("rag")
        
        # 解析各查询语句
        queries = [
            ("RAG_QUERY", "rag_query"),
            ("SEARCH_QUERY", "search_query"),
            ("FINANCIAL_API_QUERY", "financial_api_query"),
            ("ANNOUNCEMENT_QUERY", "announcement_query"),
            ("MARKET_DATA_QUERY", "market_data_query")
        ]
        
        for pattern, key in queries:
            match = re.search(rf'{pattern}:\s*(.+?)(?=\n|$)', text, re.IGNORECASE | re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
        
        return result
