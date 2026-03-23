"""
Plan Agent 模块

职责：
1. 作为Graph的入口节点，接收用户输入
2. 以股票金融专家身份对用户问题进行详细分析
3. 制定股票预测所需的详细分析计划
4. 决定需要什么数据源（rag/search/both/direct）
5. 将计划添加到state的message中
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from configs.model_config import LLMClient
from agents.base_agent import BaseAgent, agent_run_wrapper, AgentTimer
from core.state import AgentState


class PlanAgent(BaseAgent):
    """
    Plan Agent - 股票金融专家分析 + 详细计划生成
    
    作为LangGraph工作流的入口节点，负责：
    - 分析用户关于股票的询问
    - 识别分析所需的关键维度
    - 制定详细的股票分析执行计划
    """
    
    def __init__(self):
        super().__init__("PlanAgent")
        self.llm_client = LLMClient()
    
    @agent_run_wrapper
    def analyze_and_plan(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        分析用户问题并生成详细的股票分析计划，同时决定需要什么数据源
        
        Args:
            question: 用户关于股票的询问
            chat_history: 历史对话记录
            
        Returns:
            包含分析报告、计划和所需数据源的字典
        """
        self._log_step("接收用户问题", f"question={question}")
        
        # 构建系统提示词 - 股票金融专家角色
        system_prompt = """你是一位资深的股票金融专家，拥有20年证券投资分析经验。

你的专业领域包括：
- 宏观经济分析与行业趋势研判
- 公司基本面分析（财务报表、商业模式、竞争优势）
- 技术面分析（K线形态、均线系统、成交量）
- 市场情绪与资金流向分析
- 政策面与消息面解读
- 风险评估与投资组合管理

你的分析风格严谨、逻辑清晰，善于从多维度进行深度剖析。

重要：你需要根据问题的性质，决定需要什么数据源来回答。"""

        # 构建分析提示词
        history_text = ""
        if chat_history:
            history_text = "\n\n历史对话记录：\n"
            for msg in chat_history[-5:]:  # 只取最近5轮
                role = msg.get("role", "")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
        
        analysis_prompt = f"""请对用户关于股票的询问进行专业分析，制定详细的分析计划，并决定需要什么数据源。

用户问题：{question}{history_text}

## 可用数据源说明
1. **知识库(rag)**: 包含股票基础知识、历史财报数据、经典投资理论、已收录的公司分析
2. **实时搜索(search)**: 获取最新股价、新闻动态、市场情绪、政策变化等实时信息

请按以下结构输出：

## 📊 问题分析
1. **问题类型识别**：判断用户询问属于哪种类型（个股分析、行业分析、大盘走势、投资策略等）
2. **涉及标的**：明确用户关心的股票/基金代码或公司名称
3. **关注重点**：分析用户最关心的维度（价格预测、风险评估、买入时机、长期价值等）
4. **隐含需求**：推测用户可能未明确表达但需要的分析内容

## 📋 详细分析计划
请制定一个完整的股票分析执行计划，包含以下步骤（根据问题类型灵活调整）：

### 阶段一：信息收集
- [ ] 公司基本面信息获取（财报、业务模式、管理层）
- [ ] 行业动态与竞争格局调研
- [ ] 宏观经济与政策环境分析
- [ ] 近期市场消息与舆情监控

### 阶段二：技术分析
- [ ] 股价走势与K线形态分析
- [ ] 技术指标计算（MA、MACD、RSI、布林带等）
- [ ] 成交量与资金流向分析
- [ ] 支撑/阻力位识别

### 阶段三：基本面分析
- [ ] 财务报表分析（营收、利润、现金流）
- [ ] 估值分析（PE、PB、PEG、DCF）
- [ ] 盈利能力与成长性评估
- [ ] 资产负债与偿债能力分析

## 🎯 数据源决策与查询语句
基于上述分析，请决定需要什么数据源，并编写具体的查询语句。

### 决策逻辑：
- 如果问题涉及**最新股价、今天行情、新闻、政策变化** → 需要 search
- 如果问题涉及**公司基本面、历史财报、经典理论** → 需要 rag  
- 如果问题**既需要历史数据又需要最新动态** → 需要 rag + search
- 如果问题**非常简单或纯理论** → 可以都不需要（后续直接回答）

### 查询语句要求：
- **RAG_QUERY**: 如果需要知识库检索，请编写针对知识库的查询语句（用于向量检索）
- **SEARCH_QUERY**: 如果需要联网搜索，请编写针对搜索引擎的查询语句（用于 web search）

**重要提示**：当前时间是 {datetime.now().strftime('%Y年%m月')}。搜索"今天"、"现在"、"最新"等时效性信息时，请根据当前时间自行判断是否需要添加年份，通常添加当前年份有助于获取更准确的结果。

请按以下格式输出决策和查询语句：

```
REQUIRED_SOURCES: ["rag"] 或 ["search"] 或 ["rag", "search"] 或 []
RAG_QUERY: <知识库查询语句，如"贵州茅台2023年财务报表 净利润 营收">
SEARCH_QUERY: <搜索查询语句，如"贵州茅台股票今日行情 最新消息">
```

示例1（仅需知识库）：
```
REQUIRED_SOURCES: ["rag"]
RAG_QUERY: 贵州茅台 基本面分析 财务指标 PE估值
SEARCH_QUERY: 
```

示例2（仅需搜索）：
```
REQUIRED_SOURCES: ["search"]
RAG_QUERY: 
SEARCH_QUERY: 贵州茅台股票今日行情 股价走势 最新消息
```

示例3（两者都需要）：
```
REQUIRED_SOURCES: ["rag", "search"]
RAG_QUERY: 贵州茅台 财报数据 盈利能力 ROE 毛利率
SEARCH_QUERY: 贵州茅台今日股价 新闻动态 市场表现
```

示例4（都不需要）：
```
REQUIRED_SOURCES: []
RAG_QUERY: 
SEARCH_QUERY: 
```

请用专业、清晰的格式输出上述分析，并在最后严格按格式给出数据源决策和查询语句。"""

        with AgentTimer("专家分析与计划生成", self.logger):
            expert_analysis = self.llm_client.generate(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                temperature=0.5
            )
        
        self.logger.info(f"生成分析与计划，长度: {len(expert_analysis)} 字符")
        
        # 解析 LLM 的回复，提取结构化信息
        parsed = self._parse_structured_output(expert_analysis)
        required_sources = parsed["required_sources"]
        rag_query = parsed["rag_query"]
        search_query = parsed["search_query"]
        
        self.logger.info(f"数据源决策: {required_sources}")
        self.logger.info(f"RAG查询: {rag_query[:50] if rag_query else '无'}...")
        self.logger.info(f"搜索查询: {search_query[:50] if search_query else '无'}...")
        
        return {
            "analysis": expert_analysis,
            "required_sources": required_sources,
            "rag_query": rag_query,
            "search_query": search_query,
            "question": question,
            "plan_added": True
        }
    
    def _parse_structured_output(self, analysis_text: str) -> Dict[str, Any]:
        """
        从 LLM 的分析报告中解析结构化的输出
        
        Args:
            analysis_text: LLM 生成的分析报告
            
        Returns:
            包含 required_sources, rag_query, search_query 的字典
        """
        result = {
            "required_sources": [],
            "rag_query": "",
            "search_query": ""
        }
        
        # 解析 REQUIRED_SOURCES
        import re
        
        # 匹配 REQUIRED_SOURCES: [...]
        sources_match = re.search(r'REQUIRED_SOURCES:\s*(\[[^\]]*\])', analysis_text, re.IGNORECASE)
        if sources_match:
            sources_str = sources_match.group(1)
            # 解析列表
            if '"rag"' in sources_str or "'rag'" in sources_str:
                result["required_sources"].append("rag")
            if '"search"' in sources_str or "'search'" in sources_str:
                result["required_sources"].append("search")
        
        # 如果没匹配到，用关键词辅助判断
        if not result["required_sources"]:
            text_lower = analysis_text.lower()
            realtime_keywords = ["最新", "今天", "现在", "当前", "行情", "实时", "新闻", "动态"]
            need_realtime = any(kw in text_lower for kw in realtime_keywords)
            history_keywords = ["历史", "基本面", "财报", "理论", "估值", "分析"]
            need_history = any(kw in text_lower for kw in history_keywords)
            
            if need_realtime and need_history:
                result["required_sources"] = ["rag", "search"]
            elif need_realtime:
                result["required_sources"] = ["search"]
            elif need_history:
                result["required_sources"] = ["rag"]
        
        # 解析 RAG_QUERY
        rag_match = re.search(r'RAG_QUERY:\s*(.+?)(?=\n|SEARCH_QUERY:|$)', analysis_text, re.IGNORECASE | re.DOTALL)
        if rag_match:
            result["rag_query"] = rag_match.group(1).strip()
        
        # 解析 SEARCH_QUERY
        search_match = re.search(r'SEARCH_QUERY:\s*(.+?)(?=\n|```|$)', analysis_text, re.IGNORECASE | re.DOTALL)
        if search_match:
            result["search_query"] = search_match.group(1).strip()
        
        # 如果没有提取到查询语句但需要使用该数据源，则使用原始问题
        if "rag" in result["required_sources"] and not result["rag_query"]:
            result["rag_query"] = analysis_text.split('\n')[0] if analysis_text else ""
        if "search" in result["required_sources"] and not result["search_query"]:
            result["search_query"] = analysis_text.split('\n')[0] if analysis_text else ""
        
        return result
    
    @classmethod
    def as_node(cls):
        """
        创建 LangGraph 节点函数
        
        作为Graph的入口节点，将专家分析和详细计划添加到state的message中
        
        Returns:
            LangGraph 节点函数
        """
        agent = cls()
        
        def chat_node(state: AgentState) -> Dict[str, Any]:
            """
            Chat Agent 节点函数
            
            输入state包含：
            - question: 用户问题
            - chat_history: 对话历史（可选）
            - messages: 消息列表（如果没有则创建）
            
            输出更新：
            - 将专家分析和计划添加到messages中
            """
            question = state.get("question", "")
            if not question:
                return {"messages": ["未收到用户问题"]}
            
            chat_history = state.get("chat_history", [])
            
            # 执行专家分析和计划生成
            result = agent.analyze_and_plan(
                question=question,
                chat_history=chat_history
            )
            
            # 构建要添加到messages的内容
            plan_message = f"""【股票金融专家分析】

{result["analysis"]}

---
🤖 我是您的股票分析助手，已为您制定上述详细分析计划。接下来我将按照计划逐步执行分析，请稍候..."""
            
            # 获取现有的messages列表，如果没有则创建
            messages = state.get("messages", [])
            if messages is None:
                messages = []
            
            # 添加用户问题和专家分析计划
            messages.append(f"用户问题：{question}")
            messages.append(plan_message)
            
            return {
                "messages": messages,
                "analysis_plan": result["analysis"],
                "required_sources": result["required_sources"],
                "rag_query": result["rag_query"],
                "search_query": result["search_query"],
                "plan_ready": True
            }
        
        return chat_node


