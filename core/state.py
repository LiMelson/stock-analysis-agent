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
    
    # ========== 意图分类结果 ==========
    intent_type: str                        # 意图类型: real_time_quote|historical_report|investment_analysis|general_question|greeting
    intent_confidence: float                # 意图分类置信度
    intent_reasoning: str                   # 意图分类理由
    
    # ========== 需求分析结果 ==========
    requirement_analysis: Dict[str, Any]    # 需求分析结果
    
    # ========== PlanAgent 生成 ==========
    analysis_plan: str                      # 详细分析计划（完整文本）
    required_sources: List[str]             # 需要的数据源: ["index"], ["sentiment"], ["theme"], ["stock"], 或组合
    plan_ready: bool                        # 计划是否已生成
    
    # 结构化的查询计划
    index_query: str                        # 指数查询语句
    sentiment_query: str                    # 情绪查询语句
    theme_query: str                        # 题材查询语句
    stock_query: str                        # 个股查询语句
    
    # ========== 各数据源结果 ==========
    # RAG 知识库
    rag_result: Optional[str]               # RAG 检索结果（默认为空）
    rag_sources: List[Dict[str, Any]]       # RAG 来源（默认为空）
    
    # 指数数据
    index_result: Optional[str]             # 指数分析结果（默认为空）
    index_sources: List[Dict[str, Any]]     # 指数数据来源（默认为空）
    
    # 情绪数据
    sentiment_result: Optional[str]         # 情绪分析结果（默认为空）
    sentiment_sources: List[Dict[str, Any]] # 情绪数据来源（默认为空）
    
    # 题材数据
    theme_result: Optional[str]             # 题材分析结果（默认为空）
    theme_sources: List[Dict[str, Any]]     # 题材数据来源（默认为空）
    
    # 个股数据
    stock_result: Optional[str]             # 个股分析结果（默认为空）
    stock_sources: List[Dict[str, Any]]     # 个股数据来源（默认为空）
    
    # ========== 数据溯源信息 ==========
    # 用于报告可解释性，记录每个信息片段的来源
    source_attributions: List[Dict[str, Any]]  # 信息来源归因列表
    
    # ========== 最终结果 ==========
    final_answer: str                       # SummaryAgent 生成的最终投资建议报告
    has_report: bool                        # 是否已生成报告
