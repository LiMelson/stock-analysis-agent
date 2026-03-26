"""
Graph 构建模块

构建完整的股票分析工作流，包含：
- PlanAgent: 分析用户需求，生成计划，决定数据源
- IndexAgent: 大盘指数数据
- StockAgent: 个股数据
- SentimentAgent: 市场情绪数据
- ThemeAgent: 题材板块数据
- RAGAgent: 知识库检索
- SearchAgent: 联网搜索
- SummaryAgent: 综合分析，生成最终报告
- CleanupNode: 清理状态，保留对话历史
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state import AgentState
from core.nodes import (
    create_summary_node,
    create_cleanup_node,
    create_index_node,
    create_stock_node,
    create_sentiment_node,
    create_theme_node,
)
from core.router import route_decision
from agents import PlanAgent


def build_graph(rag_tool=None, checkpointer=None):
    """
    构建股票分析工作流（支持多数据源并行执行）
    
    执行流程：
    - plan_agent 分析需求，决定需要哪些数据源（index/stock/sentiment/theme/rag/search）
    - 根据 required_sources 并行执行对应的数据源节点
    - 所有数据源节点完成后汇聚到 summary
    - summary 使用知识库 + 所有数据源结果生成最终报告
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
    
    # 数据源节点：大盘指数
    workflow.add_node("index_agent", create_index_node())
    
    # 数据源节点：个股数据
    workflow.add_node("stock_agent", create_stock_node())
    
    # 数据源节点：市场情绪
    workflow.add_node("sentiment_agent", create_sentiment_node())
    
    # 数据源节点：题材板块
    workflow.add_node("theme_agent", create_theme_node())
    
    # SummaryAgent: 综合分析，生成最终报告（在节点内部使用知识库）
    workflow.add_node("summary", create_summary_node())
    
    # CleanupNode: 清理中间状态，保留对话历史
    workflow.add_node("cleanup", create_cleanup_node(max_history_turns=10))
    
    # ========== 2. 设置入口 ==========
    workflow.set_entry_point("plan_agent")
    
    # ========== 3. 添加条件路由 ==========
    
    # PlanAgent 后的路由决策 - 根据 required_sources 决定执行哪些数据源节点
    # 支持并行执行：多个数据源节点同时启动
    workflow.add_conditional_edges(
        "plan_agent", 
        route_decision, 
        ["index_agent", "stock_agent", "sentiment_agent", "theme_agent", "summary"]
    )
    
    # ========== 4. 添加汇聚边 ==========
    
    # 所有数据源节点都连接到 summary
    # 注意：当多个节点并行执行时，LangGraph 会等它们都完成后再执行 summary
    workflow.add_edge("index_agent", "summary")
    workflow.add_edge("stock_agent", "summary")
    workflow.add_edge("sentiment_agent", "summary")
    workflow.add_edge("theme_agent", "summary")
    
    # Summary 后清理，然后结束
    workflow.add_edge("summary", "cleanup")
    workflow.add_edge("cleanup", END)
    
    # ========== 5. 编译 ==========
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


# 便捷函数：创建默认应用
def create_app():
    """
    创建默认的股票分析应用（支持并行）
    """
    return build_graph()
