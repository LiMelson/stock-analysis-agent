"""
Summary Agent 模块

职责：
1. 综合 state 中的核心分析结果（用户问题、分析计划、RAG结果、搜索结果）
2. 生成最终的股票投资建议报告
3. 提供明确的买入/持有/卖出建议及理由
"""
from typing import Dict, Any
from configs.model_config import LLMClient
from agents.base_agent import BaseAgent, agent_run_wrapper, AgentTimer
from core.state import AgentState


class SummaryAgent(BaseAgent):
    """
    Summary Agent - 综合分析 + 生成最终投资建议
    
    作为LangGraph工作流的终点节点，负责：
    - 整合核心分析结果（而非对话历史）
    - 生成专业的股票投资建议报告
    - 给出明确的操作建议和风险控制
    """
    
    def __init__(self):
        super().__init__("SummaryAgent")
        self.llm_client = LLMClient()
    
    def generate_report(
        self,
        question: str,
        analysis_plan: str = "",
        rag_answer: str = "",
        search_result: str = "",
        stream_callback=None
    ) -> Dict[str, Any]:
        """
        基于核心分析结果生成最终投资建议报告（流式输出）
        
        Args:
            question: 用户问题
            analysis_plan: ChatAgent生成的分析计划
            rag_answer: RAG检索结果
            search_result: 网络搜索结果
            stream_callback: 流式输出回调函数
            
        Returns:
            包含最终报告和建议的字典
        """
        self._log_step("开始生成最终报告", f"问题: {question}")
        
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
- 给出具体的操作策略（买入价位、目标价位、止损位）"""

        # 构建信息汇总（只包含核心分析结果，不包含对话历史）
        info_sections = []
        
        if analysis_plan:
            info_sections.append(f"【分析计划】\n{analysis_plan}")
        
        if rag_answer:
            info_sections.append(f"【知识库检索结果】\n{rag_answer}")
        
        if search_result:
            info_sections.append(f"【网络搜索结果】\n{search_result}")
        
        info_text = "\n\n" + "\n\n".join(info_sections) if info_sections else "暂无其他分析结果。"
        
        # 构建开放式提示词，让模型智能决定输出格式
        summary_prompt = f"""你是一位资深的股票投资顾问。请根据用户的问题类型，灵活地组织回答内容。

用户问题：{question}

收集到的分析信息：
{info_text}

请根据问题的性质，智能决定回答方式：
- 如果问的是**概念/理论**（如"什么是市盈率"）：给出清晰简洁的概念解释，不需要投资建议格式
- 如果问的是**个股分析**（如"分析一下茅台"）：给出全面的投资分析报告
- 如果问的是**市场走势**（如"大盘怎么样"）：给出市场研判和策略建议
- 如果问的是**操作策略**（如"现在能买入吗"）：直接给出明确的投资建议

请灵活调整输出结构和深度，以最符合用户需求的方式回答。不需要拘泥于固定格式。"""

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
            "has_report": True
        }
    
    @classmethod
    def as_node(cls):
        """
        创建 LangGraph 节点函数（流式输出）
        
        Returns:
            LangGraph 节点函数
        """
        agent = cls()
        
        def summary_node(state: AgentState) -> Dict[str, Any]:
            """Summary Agent 节点函数"""
            question = state.get("question", "")
            analysis_plan = state.get("analysis_plan", "")
            rag_answer = state.get("rag_answer", "")
            search_result = state.get("search_result", "")
            
            if not question:
                return {"final_answer": "未收到用户问题，无法生成报告。", "has_report": False}
            
            # 流式输出回调
            def stream_callback(token: str):
                print(token, end="", flush=True)
            
            # 生成最终报告（流式）
            result = agent.generate_report(
                question=question,
                analysis_plan=analysis_plan,
                rag_answer=rag_answer,
                search_result=search_result,
                stream_callback=stream_callback
            )
            
            return {
                "messages": state.get("messages", []) + ["【分析报告已生成】"],
                "final_answer": result["final_answer"],
                "has_report": True
            }
        
        return summary_node
