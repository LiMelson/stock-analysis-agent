"""
Router 模块

控制 LangGraph workflow 的路由决策。
根据 PlanAgent 生成的 required_sources 决定执行路径，支持并行执行。
"""

from langgraph.constants import Send
from core.state import AgentState


def route_decision(state: AgentState):
    """
    路由决策函数 - 支持并行执行
    
    根据 PlanAgent 设置的 required_sources 决定路由：
    - [Send("rag_agent")]: 仅知识库
    - [Send("search_agent")]: 仅搜索
    - [Send("rag_agent"), Send("search_agent")]: RAG 和 Search 并行执行
    - [Send("summary")]: 直接总结（不需要数据源）
    
    使用示例:
        workflow.add_conditional_edges(
            "plan_agent",
            route_decision,
            ["rag_agent", "search_agent", "summary"]
        )
    
    Args:
        state: 当前状态，包含 required_sources 字段
        
    Returns:
        Send 对象或 Send 对象列表，表示要执行的路径
    """
    required = state.get("required_sources", [])
    sources = set(s.lower() for s in required)
    
    has_rag = "rag" in sources
    has_search = "search" in sources
    
    if has_rag and has_search:
        # 并行执行 RAG 和 Search
        return [Send("rag_agent", state), Send("search_agent", state)]
    elif has_rag:
        return Send("rag_agent", state)
    elif has_search:
        return Send("search_agent", state)
    else:
        return Send("summary", state)
