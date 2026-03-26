"""
Plan Agent 模块

整合以下子模块作为Graph的入口节点：
- IntentRequirementAgent: 意图分类与需求理解（合并模块）
- DataSourceRouter: 数据源路由决策

职责：
1. 作为Graph的入口节点，接收用户输入
2. 调用子模块完成意图分析、需求理解和数据源路由
3. 生成分析计划并写入state
"""

from typing import Dict, Any, Optional, List
from ..base_agent import BaseAgent, agent_run_wrapper, AgentTimer
from .intent_requirement_agent import IntentRequirementAgent, IntentType
from .datasource_router import DataSourceRouter
from core.state import AgentState


class PlanAgent(BaseAgent):
    """
    Plan Agent - 整合意图需求分析和数据源路由
    
    作为LangGraph工作流的入口节点，负责：
    1. 意图分类与需求理解（合并为一步）
    2. 数据源路由决策
    3. 生成分析计划
    """
    
    def __init__(self):
        super().__init__("PlanAgent")
        self.intent_requirement_agent = IntentRequirementAgent()
        self.datasource_router = DataSourceRouter()
    
    @agent_run_wrapper
    def analyze_and_plan(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        完整的分析与计划流程
        
        Args:
            question: 用户问题
            chat_history: 历史对话记录
            
        Returns:
            包含完整分析结果的字典
        """
        self._log_step("开始完整分析流程", f"question={question}")
        
        # ===== 步骤1: 意图与需求分析（合并） =====
        analysis_result = self.intent_requirement_agent.analyze(question, chat_history)
        intent_type = analysis_result.get("intent_type", "unknown")
        
        # ===== 步骤2: 数据源路由 =====
        routing_result = self.datasource_router.route(
            question=question,
            intent_type=intent_type,
            requirement_analysis=analysis_result
        )
        
        # ===== 步骤3: 生成详细分析计划 =====
        analysis_plan = self._generate_analysis_plan(
            question=question,
            analysis_result=analysis_result,
            routing_result=routing_result
        )
        
        self.logger.info(f"分析完成 - 意图: {intent_type}, 数据源: {routing_result['required_sources']}")
        
        return {
            "analysis": analysis_plan,
            "intent_type": intent_type,
            "intent_confidence": analysis_result.get("intent_confidence"),
            "intent_reasoning": analysis_result.get("intent_reasoning"),
            "requirement_analysis": analysis_result,
            "required_sources": routing_result["required_sources"],
            "index_query": routing_result.get("index_query", ""),
            "sentiment_query": routing_result.get("sentiment_query", ""),
            "theme_query": routing_result.get("theme_query", ""),
            "stock_query": routing_result.get("stock_query", ""),
            "rag_query": routing_result.get("rag_query", ""),
            "question": question,
            "plan_added": True
        }
    
    def _generate_analysis_plan(
        self,
        question: str,
        analysis_result: Dict[str, Any],
        routing_result: Dict[str, Any]
    ) -> str:
        """生成完整的分析计划文本"""
        
        intent_type = analysis_result.get("intent_type", "unknown")
        intent_confidence = analysis_result.get("intent_confidence", 0.5)
        intent_reasoning = analysis_result.get("intent_reasoning", "")
        
        intent_type_display = {
            "real_time_quote": "实时行情查询",
            "historical_report": "历史研报查询",
            "investment_analysis": "投资分析",
            "general_question": "一般问题",
            "greeting": "问候",
            "unknown": "未知意图"
        }.get(intent_type, intent_type)
        
        target_stock = analysis_result.get("target_stock", "未识别")
        focus_areas = analysis_result.get("focus_areas", [])
        question_type = analysis_result.get("question_type", "未识别")
        implicit_needs = analysis_result.get("implicit_needs", "无")
        
        required_sources = routing_result["required_sources"]
        sources_display = []
        for s in required_sources:
            if s == "rag":
                sources_display.append("知识库检索")
            elif s == "search":
                sources_display.append("实时搜索")
            elif s == "financial_api":
                sources_display.append("金融数据API（预留）")
            elif s == "announcement_db":
                sources_display.append("公告数据库（预留）")
            elif s == "market_data":
                sources_display.append("市场行情数据（预留）")
        
        plan = f"""## 📊 意图识别

**意图类型**: {intent_type_display}
**置信度**: {intent_confidence:.2f}
**分类理由**: {intent_reasoning}

## 📋 需求分析

1. **问题类型**: {question_type}
2. **涉及标的**: {target_stock}
3. **关注重点**: {', '.join(focus_areas) if focus_areas else '未明确'}
4. **隐含需求**: {implicit_needs}

## 🎯 数据源决策

**选用数据源**: {', '.join(sources_display) if sources_display else '无需外部数据'}

### 查询计划：
"""
        
        if routing_result.get("rag_query"):
            plan += f"- **知识库查询**: {routing_result['rag_query']}\n"
        if routing_result.get("search_query"):
            plan += f"- **搜索查询**: {routing_result['search_query']}\n"
        if routing_result.get("financial_api_query"):
            plan += f"- **金融API查询**（预留）: {routing_result['financial_api_query']}\n"
        if routing_result.get("announcement_query"):
            plan += f"- **公告查询**（预留）: {routing_result['announcement_query']}\n"
        
        plan += f"""
### 执行计划：
"""
        
        if "rag" in required_sources:
            plan += "- [ ] 知识库检索 - 获取历史财报和基本面数据\n"
        if "search" in required_sources:
            plan += "- [ ] 实时搜索 - 获取最新股价和市场动态\n"
        if "financial_api" in required_sources:
            plan += "- [ ] 金融API查询（预留）- 获取结构化财务数据\n"
        if "announcement_db" in required_sources:
            plan += "- [ ] 公告查询（预留）- 获取公司公告信息\n"
        
        plan += "- [ ] 综合分析 - 整合多源数据生成投资建议\n"
        
        return plan
    
    @classmethod
    def as_node(cls):
        """
        创建 LangGraph 节点函数
        
        Returns:
            LangGraph 节点函数
        """
        agent = cls()
        
        def chat_node(state: AgentState) -> Dict[str, Any]:
            """
            Plan Agent 节点函数
            
            输入state包含：
            - question: 用户问题
            - chat_history: 对话历史（可选）
            
            输出更新：
            - 将专家分析和计划添加到messages中
            """
            question = state.get("question", "")
            if not question:
                return {"messages": ["未收到用户问题"]}
            
            chat_history = state.get("chat_history", [])
            
            print(f"    [PlanAgent] 正在分析意图...", flush=True)
            
            # 执行完整分析流程
            result = agent.analyze_and_plan(
                question=question,
                chat_history=chat_history
            )
            
            print(f"    [PlanAgent] 意图识别: {result['intent_type']}, 数据源: {result['required_sources']}", flush=True)
            
            # 构建要添加到messages的内容
            intent_type_display = {
                "real_time_quote": "📈 实时行情",
                "historical_report": "📚 历史研报",
                "investment_analysis": "🔍 投资分析",
                "general_question": "❓ 知识问答",
                "greeting": "👋 问候",
                "unknown": "📝 需求分析"
            }.get(result["intent_type"], "📝 需求分析")
            
            plan_message = f"""【{intent_type_display}】

{result["analysis"]}

---
🤖 我是您的股票分析助手，已识别您的意图并制定分析计划。接下来将执行数据收集和分析，请稍候..."""
            
            # 获取现有的messages列表
            messages = state.get("messages", [])
            if messages is None:
                messages = []
            
            # 添加用户问题和分析计划
            messages.append(f"用户问题：{question}")
            messages.append(plan_message)
            
            return {
                "messages": messages,
                "analysis_plan": result["analysis"],
                "intent_type": result["intent_type"],
                "intent_confidence": result["intent_confidence"],
                "intent_reasoning": result["intent_reasoning"],
                "requirement_analysis": result["requirement_analysis"],
                "required_sources": result["required_sources"],
                "index_query": result.get("index_query", ""),
                "sentiment_query": result.get("sentiment_query", ""),
                "theme_query": result.get("theme_query", ""),
                "stock_query": result.get("stock_query", ""),
                "rag_query": result.get("rag_query", ""),
                "plan_ready": True
            }
        
        return chat_node
