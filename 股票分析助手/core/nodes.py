from typing import Optional, Callable, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from rag.rag_retriever import RAGTool


def create_rag_node(
    rag_tool: Optional['RAGTool'] = None,
    output_key: str = "rag_answer",
    top_k: int = 5,
    use_mqe: bool = False,
    use_hyde: bool = False
) -> Callable:
    """
    创建 RAG 节点（检索 + 专家分析总结）
    
    功能：从 state 取 rag_query（PlanAgent 生成）-> RAGAgent 检索并分析总结 -> 结果写入 state
    
    注意：优先使用 state.rag_query，如果没有则使用 state.question
    如果 rag_tool 为 None，返回一个返回空结果的节点
    """
    if rag_tool is None:
        # 如果没有 RAG 工具，返回空结果
        def empty_rag_node(state) -> Dict[str, Any]:
            return {output_key: "未配置知识库"}
        return empty_rag_node
    
    from agents.rag_agent import RAGAgent
    return RAGAgent.as_node(
        rag_tool=rag_tool,
        output_key=output_key,
        top_k=top_k,
        use_mqe=use_mqe,
        use_hyde=use_hyde
    )


def create_search_node(
    api_key: Optional[str] = None,
    max_results: int = 5,
    update_key: str = "search_result"
) -> Callable:
    """
    创建网络搜索节点（搜索 + 专家分析总结）
    
    功能：从 state 取问题 -> SearchAgent 搜索并分析总结 -> 结果写入 state
    
    如果未配置搜索 API，返回一个返回提示信息的节点
    """
    import os
    
    # 检查是否有搜索 API key
    has_api_key = api_key is not None or os.getenv("TAVILY_API_KEY") is not None
    
    if not has_api_key:
        # 如果没有 API key，返回提示信息
        def empty_search_node(state) -> Dict[str, Any]:
            return {update_key: "未配置搜索 API（请设置 TAVILY_API_KEY 环境变量）"}
        return empty_search_node
    
    from agents.search_agent import SearchAgent
    return SearchAgent.as_node(
        api_key=api_key,
        max_results=max_results,
        update_key=update_key
    )


def create_plan_node() -> Callable:
    """
    创建 Plan 节点（股票金融专家分析 + 详细计划生成 + 数据源决策）
    
    功能：从 state 取用户问题 -> PlanAgent 进行专家分析 -> 
          制定详细分析计划 + 决定需要什么数据源 -> 写入 state
    
    作为 Graph 的入口节点，负责：
    - 识别用户问题的类型和关注重点
    - 制定股票预测所需的详细分析计划
    - 决定需要什么数据源（rag/search/both/direct）
    - 输出结构化的查询计划（rag_query, search_query）
    - 将计划和数据源决策写入 state 供 Router 使用
    
    Returns:
        LangGraph 节点函数
    
    使用示例:
        from agents import PlanAgent
        workflow.add_node("plan", PlanAgent.as_node())
        
        或:
        from core.nodes import create_plan_node
        workflow.add_node("plan", create_plan_node())
    """
    from agents.plan_agent import PlanAgent
    return PlanAgent.as_node()


def create_summary_node() -> Callable:
    """
    创建总结节点（流式输出）
    
    Returns:
        LangGraph 节点函数
    """
    from agents.summary_agent import SummaryAgent
    return SummaryAgent.as_node()


def create_cleanup_node(max_history_turns: int = 10) -> Callable:
    """创建状态清理节点 - 保留对话历史，清理中间状态"""
    from core.state import AgentState
    
    def cleanup_node(state: AgentState) -> Dict[str, Any]:
        """清理中间状态，保留对话历史用于多轮对话"""
        question = state.get("question", "")
        final_answer = state.get("final_answer", "")
        
        # 只更新 chat_history（累积历史，给 PlanAgent 做上下文）
        chat_history = state.get("chat_history", []) or []
        if question:
            chat_history.append({"role": "user", "content": question})
        if final_answer:
            chat_history.append({"role": "assistant", "content": final_answer})
        
        # 限制历史长度（只保留最近 N 轮）
        max_messages = max_history_turns * 2
        if len(chat_history) > max_messages:
            chat_history = chat_history[-max_messages:]
        
        # 只返回 chat_history，其他字段（messages, analysis_plan, rag_answer等）自动清理
        return {"chat_history": chat_history}
    
    return cleanup_node
