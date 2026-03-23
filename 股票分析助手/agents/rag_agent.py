"""
RAG Agent 模块

职责：
1. 从知识库检索相关信息
2. 对检索结果进行专家式分析和总结
3. 生成专业回答
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from configs.model_config import LLMClient
from agents.base_agent import BaseAgent, agent_run_wrapper, AgentTimer

if TYPE_CHECKING:
    from rag.rag_retriever import RAGTool


class RAGAgent(BaseAgent):
    """
    RAG Agent - 检索 + 专家式分析总结
    """
    
    def __init__(self, rag_tool: 'RAGTool'):
        super().__init__("RAGAgent")
        self.rag_tool = rag_tool
        self.llm_client = LLMClient()
    
    @agent_run_wrapper
    def run(
        self,
        question: str,
        top_k: int = 5,
        use_mqe: bool = False,
        use_hyde: bool = False
    ) -> str:
        """
        执行 RAG 流程：检索 + 专家分析总结
        
        Args:
            question: 用户问题
            top_k: 检索结果数量
            use_mqe: 是否启用多查询扩展
            use_hyde: 是否启用假设文档嵌入
            
        Returns:
            专家式分析总结
        """
        # 1. 检索
        self._log_step("检索", f"question={question}, use_mqe={use_mqe}, use_hyde={use_hyde}")
        
        with AgentTimer("检索阶段", self.logger):
            retrieval_result = self.rag_tool.query(
                question=question,
                top_k=top_k,
                use_mqe=use_mqe,
                use_hyde=use_hyde,
                return_sources=True
            )
        
        context = retrieval_result.get("context", "")
        sources = retrieval_result.get("sources", [])
        
        if not context:
            self.logger.warning("未在知识库中找到相关信息")
            return "未在知识库中找到相关信息。"
        
        self.logger.info(f"检索到 {len(sources)} 个相关文档")
        
        # 2. 构建来源信息
        sources_info = "\n".join([
            f"[来源 {i+1}] 文档: {s.get('document_id', '未知')}, 相关度: {s.get('score', 0):.3f}"
            for i, s in enumerate(sources)
        ])
        
        # 3. 专家式分析
        self._log_step("生成专家回答")
        
        prompt = f"""你是一位专业的投资分析师。请基于以下知识库内容，对用户的问题进行深入分析和专业回答。

用户问题：{question}

检索到的知识库内容：
{context}

来源信息：
{sources_info}

请给出专业、深入的分析回答。如果检索内容中存在矛盾或不一致之处，请指出并分析原因。回答应结构清晰，有理有据。"""
        
        # 4. LLM 生成专家回答
        with AgentTimer("LLM生成", self.logger):
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是一位专业的投资分析师，擅长基于资料进行深入分析和给出权威见解。",
                temperature=0.3
            )
        
        self.logger.info(f"生成回答长度: {len(answer)} 字符")
        
        return answer
    
    @classmethod
    def as_node(
        cls,
        rag_tool: 'RAGTool',
        output_key: str = "rag_answer",
        top_k: int = 5,
        use_mqe: bool = False,
        use_hyde: bool = False
    ):
        """
        创建 LangGraph 节点函数
        
        优先使用 state 中的 rag_query（由 PlanAgent 生成），
        如果没有则使用 question。
        
        Args:
            rag_tool: RAGTool 实例
            output_key: 回答字段名
            top_k: 检索结果数量
            use_mqe: 是否启用 MQE
            use_hyde: 是否启用 HyDE
            
        Returns:
            LangGraph 节点函数
        """
        from core.state import AgentState
        agent = cls(rag_tool)
        
        def rag_node(state: AgentState) -> Dict[str, Any]:
            """RAG 节点函数
            
            输入：
            - state.rag_query: PlanAgent 生成的 RAG 查询（优先使用）
            - state.question: 用户原始问题（备选）
            
            输出：
            - rag_answer: 检索和分析结果
            """
            # 优先使用 PlanAgent 生成的 rag_query
            query = state.get("rag_query", "")
            if not query:
                # 如果没有 rag_query，使用原始问题
                query = state.get("question", "")
            
            if not query:
                return {output_key: "未提供查询内容"}
            
            agent.logger.info(f"RAG查询: {query[:100]}...")
            
            # 检索并生成专家总结
            answer = agent.run(
                question=query,
                top_k=top_k,
                use_mqe=use_mqe,
                use_hyde=use_hyde
            )
            
            return {output_key: answer}
        
        return rag_node
