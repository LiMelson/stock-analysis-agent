"""
Search Agent 模块

职责：
1. 执行网络搜索
2. 使用 LLM 对搜索结果进行专家式总结
3. 将总结加入 state
"""

from typing import Dict, Any, Optional
from tools.web_search import WebSearchTool
from configs.model_config import LLMClient
from agents.base_agent import BaseAgent, agent_run_wrapper, AgentTimer
from core.state import AgentState


class SearchAgent(BaseAgent):
    """
    搜索 Agent - 搜索 + LLM 总结
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("SearchAgent")
        self.search_tool = WebSearchTool(api_key=api_key)
        self.llm_client = LLMClient()
    
    @agent_run_wrapper
    def run(
        self,
        question: str,
        max_results: int = 5
    ) -> str:
        """
        执行搜索并生成专家式总结
        
        Args:
            question: 搜索查询
            max_results: 搜索结果数量
            
        Returns:
            LLM 生成的专家总结
        """
        # 1. 搜索
        self._log_step("网络搜索", f"query={question}, max_results={max_results}")
        
        with AgentTimer("搜索阶段", self.logger):
            contents = self.search_tool.search_contents(
                query=question,
                max_results=max_results
            )
        
        if not contents:
            self.logger.warning("未找到相关信息")
            return "未找到相关信息。"
        
        self.logger.info(f"搜索到 {len(contents)} 个结果")
        
        # 2. 构建总结提示词
        sources_text = "\n\n".join([
            f"[来源 {i+1}]\n{content}"
            for i, content in enumerate(contents)
        ])
        
        prompt = f"""你是一位专业的金融分析师。请基于以下搜索结果，对用户的问题给出专业、简洁的总结。

用户问题：{question}

搜索结果：
{sources_text}

请给出专业的总结回答，如果信息之间存在矛盾，请指出并分析。回答应简洁有力，突出重点。"""
        
        # 3. LLM 总结
        self._log_step("生成专家总结")
        
        with AgentTimer("LLM生成", self.logger):
            summary = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是一位专业的金融分析师，擅长整合多方信息并给出权威见解。",
                temperature=0.3
            )
        
        self.logger.info(f"生成总结长度: {len(summary)} 字符")
        
        return summary
    
    @classmethod
    def as_node(
        cls,
        api_key: Optional[str] = None,
        max_results: int = 5,
        update_key: str = "search_result"
    ):
        """
        创建 LangGraph 节点函数
        
        优先使用 state 中的 search_query（由 PlanAgent 生成），
        如果没有则使用 question。
        
        Args:
            api_key: Tavily API Key
            max_results: 最大结果数量
            update_key: 更新到 state 中的字段名
            
        Returns:
            LangGraph 节点函数
        """
        agent = cls(api_key=api_key)
        
        def search_node(state: AgentState) -> Dict[str, Any]:
            """LangGraph 节点函数
            
            输入：
            - state.search_query: PlanAgent 生成的搜索查询（优先使用）
            - state.question: 用户原始问题（备选）
            
            输出：
            - search_result: 搜索结果总结
            """
            # 优先使用 PlanAgent 生成的 search_query
            query = state.get("search_query", "")
            if not query:
                # 如果没有 search_query，使用原始问题
                query = state.get("question", "")
            
            if not query:
                return {update_key: "未提供查询内容"}
            
            agent.logger.info(f"搜索查询: {query[:100]}...")
            
            # 搜索并生成总结
            summary = agent.run(
                question=query,
                max_results=max_results
            )
            
            return {update_key: summary}
        
        return search_node
