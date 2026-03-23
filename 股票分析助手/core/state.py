"""
状态定义模块

提供 LangGraph RAG 工作流的标准状态定义。
"""

from typing import List, Dict, Any, TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    question: str                           # 用户问题
    chat_history: Annotated[List[Dict], add_messages]  # 对话历史
    messages: List[str]                     # 节点执行过程中的消息记录
    
    # PlanAgent 生成
    analysis_plan: str                      # 详细分析计划（完整文本）
    required_sources: List[str]             # 需要的数据源: ["rag"], ["search"], ["rag", "search"], []
    plan_ready: bool                        # 计划是否已生成
    
    # 结构化的查询计划（供 RAG 和 Search 节点使用）
    rag_query: str                          # RAG 检索的具体查询语句
    search_query: str                       # 联网搜索的具体查询语句
    
    # 各数据源结果
    rag_answer: str                         # 本地知识库检索结果
    search_result: Optional[str]            # 联网搜索结果
    
    # 最终结果
    final_answer: str                       # SummaryAgent 生成的最终投资建议报告
    has_report: bool                        # 是否已生成报告
