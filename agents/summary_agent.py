"""
Summary Agent 模块

职责：
1. 综合 state 中的核心分析结果（用户问题、分析计划、RAG结果、搜索结果）
2. 主动查询知识库作为判断和总结的依据
3. 生成最终的股票投资建议报告
4. 在报告中标注信息来源，提升可解释性
5. 提供明确的买入/持有/卖出建议及理由
"""
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from configs.model_config import LLMClient
from agents.base_agent import BaseAgent, agent_run_wrapper, AgentTimer
from core.state import AgentState

if TYPE_CHECKING:
    from rag.rag_retriever import RAGTool


class SummaryAgent(BaseAgent):
    """
    Summary Agent - 综合分析 + 生成最终投资建议
    
    作为LangGraph工作流的终点节点，负责：
    - 整合核心分析结果（而非对话历史）
    - 主动查询知识库作为判断依据
    - 生成专业的股票投资建议报告
    - 在报告中标注信息来源
    - 给出明确的操作建议和风险控制
    """
    
    def __init__(self, rag_tool: Optional['RAGTool'] = None):
        super().__init__("SummaryAgent")
        self.llm_client = LLMClient()
        self.rag_tool = rag_tool
    
    def _format_source_attributions(
        self,
        rag_sources: Optional[List[Dict]] = None,
        search_sources: Optional[List[Dict]] = None
    ) -> str:
        """
        格式化来源归因信息
        
        Args:
            rag_sources: RAG来源列表
            search_sources: 搜索来源列表
            
        Returns:
            格式化后的来源说明文本
        """
        sections = []
        
        if rag_sources and len(rag_sources) > 0:
            sections.append("### 📚 知识库来源")
            for s in rag_sources[:3]:  # 最多显示3个
                doc_id = s.get("document_id", "未知文档")
                score = s.get("score", 0)
                sections.append(f"- {doc_id} (相关度: {score:.3f})")
        
        if search_sources and len(search_sources) > 0:
            sections.append("### 🔍 网络搜索来源")
            for s in search_sources[:3]:  # 最多显示3个
                title = s.get("title", "未知标题")
                source = s.get("source", "网络")
                sections.append(f"- {title} (来源: {source})")
        
        return "\n".join(sections) if sections else "无外部数据来源"
    
    def _query_knowledge_base(self, question: str, top_k: int = 5) -> tuple[str, List[Dict]]:
        """
        主动查询知识库获取相关信息
        
        Args:
            question: 查询问题
            top_k: 返回结果数量
            
        Returns:
            (检索结果文本, 来源列表)
        """
        if self.rag_tool is None:
            return "", []
        
        try:
            result = self.rag_tool.query(
                question=question,
                top_k=top_k,
                return_sources=True
            )
            context = result.get("context", "")
            sources = result.get("sources", [])
            
            if not context:
                return "未在知识库中找到相关信息", []
            
            return context, sources
        except Exception as e:
            self.logger.warning(f"知识库查询失败: {e}")
            return f"知识库查询失败: {e}", []
    
    @agent_run_wrapper
    def generate_report(
        self,
        question: str,
        analysis_plan: str = "",
        rag_result: str = "",
        index_result: str = "",
        sentiment_result: str = "",
        theme_result: str = "",
        stock_result: str = "",
        search_result: str = "",
        rag_sources: Optional[List[Dict]] = None,
        search_sources: Optional[List[Dict]] = None,
        intent_type: str = "",
        stream_callback=None,
        force_rag_query: bool = True
    ) -> Dict[str, Any]:
        """
        基于核心分析结果生成最终投资建议报告（流式输出）
        
        Args:
            question: 用户问题
            analysis_plan: ChatAgent生成的分析计划
            rag_result: RAG知识库检索结果
            index_result: 指数分析结果
            sentiment_result: 情绪分析结果
            theme_result: 题材分析结果
            stock_result: 个股分析结果
            search_result: 网络搜索结果（备用）
            rag_sources: RAG来源信息
            search_sources: 搜索来源信息
            intent_type: 意图类型
            stream_callback: 流式输出回调函数
            force_rag_query: 是否强制查询知识库（默认True，Summary节点独立查询知识库）
            
        Returns:
            包含最终报告和建议的字典
        """
        self._log_step("开始生成最终报告", f"问题: {question}")
        
        # Summary节点主动查询知识库作为判断依据
        if force_rag_query and self.rag_tool is not None and not rag_result:
            self._log_step("Summary节点主动查询知识库")
            rag_result, kb_sources = self._query_knowledge_base(question, top_k=5)
            if kb_sources:
                rag_sources = (rag_sources or []) + kb_sources
        
        # 格式化来源信息
        source_info = self._format_source_attributions(rag_sources, search_sources)
        
        # 构建系统提示词 - 资深投资顾问角色
        system_prompt = """你是一位资深的股票投资顾问，拥有丰富的实战经验和卓越的业绩记录。

你的专业能力：
- 综合多方信息进行独立判断
- 平衡风险与收益，给出务实建议
- 用清晰的数据支撑观点
- 充分考虑市场不确定性和黑天鹅风险

你的建议风格：
- 客观中立，不受情绪影响
- 数据驱动，有理有据
- 风险提示充分，不夸大收益
- 给出具体的操作策略（买入价位、目标价位、止损位）
- **在回答中明确标注数据来源**，提升可信度"""

        # 构建信息汇总（只包含核心分析结果，不包含对话历史）
        info_sections = []
        data_sources_used = []
        
        if rag_result:
            info_sections.append(f"【知识库检索结果】\n{rag_result}")
            data_sources_used.append("知识库")
        
        if index_result:
            info_sections.append(f"【指数分析结果】\n{index_result}")
            data_sources_used.append("指数数据")
        
        if sentiment_result:
            info_sections.append(f"【市场情绪分析】\n{sentiment_result}")
            data_sources_used.append("情绪数据")
        
        if theme_result:
            info_sections.append(f"【题材板块分析】\n{theme_result}")
            data_sources_used.append("题材数据")
        
        if stock_result:
            info_sections.append(f"【个股数据分析】\n{stock_result}")
            data_sources_used.append("个股数据")
        
        if search_result:
            info_sections.append(f"【网络搜索结果】\n{search_result}")
            data_sources_used.append("Tavily实时搜索")
        
        if analysis_plan:
            info_sections.append(f"【分析计划】\n{analysis_plan}")
        
        info_text = "\n\n" + "\n\n".join(info_sections) if info_sections else "暂无其他分析结果。"
        
        # 根据意图类型调整输出风格
        intent_instruction = {
            "real_time_quote": "用户主要关心实时行情，重点提供最新价格、涨跌幅等数据，并标注数据来源。",
            "historical_report": "用户主要关心历史数据，重点提供财报数据、历史表现等，并标注数据来源。",
            "investment_analysis": "用户需要投资建议，提供全面分析，明确标注各数据点来源，给出操作建议。",
            "general_question": "用户询问概念知识，清晰简洁地解释，如有引用知识库内容请标注来源。",
            "greeting": "友好地回应问候。"
        }.get(intent_type, "根据问题灵活调整回答方式。")
        
        # 构建开放式提示词
        summary_prompt = f"""你是一位资深的股票投资顾问。请根据用户的问题类型，灵活地组织回答内容。

用户问题：{question}

使用的数据来源：{', '.join(data_sources_used) if data_sources_used else '无'}

收集到的分析信息：
{info_text}

来源详情：
{source_info}

回答指导：
- {intent_instruction}

请在回答中遵循以下格式：
1. **主体分析**：根据问题类型给出相应分析
2. **数据来源标注**：在涉及具体数据时，使用【来自知识库】、【来自Tavily实时搜索】等标注
3. **免责声明**：在报告末尾添加风险提示

示例格式：
```
## 分析结论
...

## 数据来源说明
- 财务数据：【来自知识库】贵州茅台2023年年报
- 实时股价：【来自Tavily实时搜索】截至X月X日

## 风险提示
...
```

请灵活调整输出结构和深度，以最符合用户需求的方式回答。"""

        # 流式输出
        print("\n【投资建议】\n", end="", flush=True)
        final_report = self.llm_client.generate(
            prompt=summary_prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=4000,
            stream=True,
            stream_callback=stream_callback
        )
        print("\n")  # 结束后换行
        
        self.logger.info(f"生成最终报告，长度: {len(final_report)} 字符")
        
        return {
            "final_answer": final_report,
            "has_report": True,
            "data_sources_used": data_sources_used
        }
    
    @classmethod
    def as_node(cls, rag_tool: Optional['RAGTool'] = None):
        """
        创建 LangGraph 节点函数（流式输出）
        
        Args:
            rag_tool: RAGTool 实例，用于Summary节点主动查询知识库
            
        Returns:
            LangGraph 节点函数
        """
        agent = cls(rag_tool=rag_tool)
        
        def summary_node(state: AgentState) -> Dict[str, Any]:
            """Summary Agent 节点函数 - 主动查询知识库作为判断依据"""
            question = state.get("question", "")
            analysis_plan = state.get("analysis_plan", "")
            rag_result = state.get("rag_result", "")
            index_result = state.get("index_result", "")
            sentiment_result = state.get("sentiment_result", "")
            theme_result = state.get("theme_result", "")
            stock_result = state.get("stock_result", "")
            search_result = state.get("search_result", "")
            rag_sources = state.get("rag_sources", [])
            search_sources = state.get("search_sources", [])
            intent_type = state.get("intent_type", "")
            
            # 收集已有溯源信息
            existing_attributions = state.get("source_attributions", [])
            
            # 合并新的溯源信息
            new_attributions = []
            if rag_sources:
                for s in rag_sources:
                    new_attributions.append({
                        "type": "knowledge_base",
                        "source": s.get("document_id", "未知"),
                        "relevance_score": s.get("score", 0),
                        "description": f"知识库文档: {s.get('document_id', '未知')}"
                    })
            if search_sources:
                for s in search_sources:
                    new_attributions.append({
                        "type": "web_search",
                        "source": s.get("url", ""),
                        "title": s.get("title", ""),
                        "description": f"Tavily搜索: {s.get('title', '未知')}"
                    })
            
            all_attributions = existing_attributions + new_attributions
            
            if not question:
                return {
                    "final_answer": "未收到用户问题，无法生成报告。",
                    "has_report": False,
                    "source_attributions": all_attributions
                }
            
            # 流式输出回调
            def stream_callback(token: str):
                print(token, end="", flush=True)
            
            # 生成最终报告（流式）- Summary节点主动查询知识库
            result = agent.generate_report(
                question=question,
                analysis_plan=analysis_plan,
                rag_result=rag_result,
                index_result=index_result,
                sentiment_result=sentiment_result,
                theme_result=theme_result,
                stock_result=stock_result,
                search_result=search_result,
                rag_sources=rag_sources,
                search_sources=search_sources,
                intent_type=intent_type,
                stream_callback=stream_callback,
                force_rag_query=True  # Summary节点主动查询知识库
            )
            
            return {
                "messages": state.get("messages", []) + ["【分析报告已生成】"],
                "final_answer": result["final_answer"],
                "has_report": True,
                "source_attributions": all_attributions,
                "data_sources_used": result.get("data_sources_used", [])
            }
        
        return summary_node
