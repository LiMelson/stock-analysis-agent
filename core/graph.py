"""
Graph 构建模块

构建完整的股票分析工作流，包含：
- PlanAgent: 分析用户需求，生成计划，决定数据源
- RAGAgent: 知识库检索（并行）
- SearchAgent: 联网搜索（并行）
- SummaryAgent: 综合分析，生成最终报告
- CleanupNode: 清理状态，保留对话历史
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from core.nodes import (
    create_rag_node,
    create_search_node,
    create_summary_node,
    create_cleanup_node,
)
from core.router import route_decision
from agents import PlanAgent


def build_graph(rag_tool=None, checkpointer=None):
    """
    构建股票分析工作流（支持并行执行）
    
    执行流程：
    - plan_agent 分析需求
    - 根据 required_sources 决定：
      * ["rag"] -> 只执行 rag_agent
      * ["search"] -> 只执行 search_agent
      * ["rag", "search"] -> rag_agent 和 search_agent 并行执行
      * [] -> 直接执行 summary
    - rag_agent 和 search_agent 执行完后都到 summary
    - summary 生成最终报告
    - cleanup 清理状态
    
    Args:
        rag_tool: RAG 检索工具实例，如果为 None 则不使用 RAG
        checkpointer: Checkpoint 存储器，默认使用 MemorySaver
        
    Returns:
        编译后的 LangGraph 应用
    """
    workflow = StateGraph(AgentState)
    
    # ========== 1. 添加节点 ==========
    
    # PlanAgent: 入口节点，分析需求，生成计划，决定数据源
    workflow.add_node("plan_agent", PlanAgent.as_node())
    
    # RAGAgent: 知识库检索（使用 plan_agent 生成的 rag_query）
    if rag_tool:
        workflow.add_node("rag_agent", create_rag_node(rag_tool))
    else:
        # 如果没有 rag_tool，创建一个空节点
        workflow.add_node("rag_agent", lambda state: {"rag_answer": "未配置知识库"})
    
    # SearchAgent: 联网搜索（使用 plan_agent 生成的 search_query）
    workflow.add_node("search_agent", create_search_node())
    
    # SummaryAgent: 综合分析，生成最终报告
    workflow.add_node("summary", create_summary_node())
    
    # CleanupNode: 清理中间状态，保留对话历史
    workflow.add_node("cleanup", create_cleanup_node(max_history_turns=10))
    
    # ========== 2. 设置入口 ==========
    workflow.set_entry_point("plan_agent")
    
    # ========== 3. 添加并行路由 ==========
    
    # PlanAgent 后的路由决策 - 根据 required_sources 决定执行路径
    # 支持并行执行：当 rag 和 search 都需要时，同时启动两个节点
    workflow.add_conditional_edges(
        "plan_agent", 
        route_decision, 
        ["rag_agent", "search_agent", "summary"]
    )
    
    # ========== 4. 添加汇聚边 ==========
    
    # rag_agent 和 search_agent 都连接到 summary
    # 注意：当两者都执行时，LangGraph 会等它们都完成后再执行 summary
    workflow.add_edge("rag_agent", "summary")
    workflow.add_edge("search_agent", "summary")
    
    # Summary 后清理，然后结束
    workflow.add_edge("summary", "cleanup")
    workflow.add_edge("cleanup", END)
    
    # ========== 5. 编译 ==========
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


# 便捷函数：创建默认应用
def create_app(rag_tool=None):
    """
    创建默认的股票分析应用（支持并行）
    """
    return build_graph(rag_tool=rag_tool)
