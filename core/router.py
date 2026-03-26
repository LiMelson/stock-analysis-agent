"""
Router 模块

控制 LangGraph workflow 的路由决策。
根据 PlanAgent 的 required_sources 决定执行哪些数据源节点。
"""

from langgraph.constants import Send
from core.state import AgentState


def route_decision(state: AgentState):
    """
    路由决策函数 - 根据 required_sources 分发到对应数据源节点
    
    路由策略：
    - PlanAgent 生成 required_sources 列表，包含需要的数据源
    - 支持并行执行多个数据源节点
    - 数据源节点包括: index(指数), stock(个股), sentiment(情绪), theme(题材), rag(知识库), search(搜索)
    - 所有数据源节点执行完成后汇聚到 summary
    
    Args:
        state: 当前状态，包含 required_sources 字段
        
    Returns:
        Send 对象或 Send 对象列表，表示要执行的路径
    """
    required = state.get("required_sources", [])
    sources = set(s.lower() for s in required)
    question = state.get("question", "").lower()
    
    # ===== 特殊情况：纯理论问题/问候，不需要数据源 =====
    no_source_keywords = ["你好", "嗨", "hello", "hi", "什么是股票", "什么是基金", "什么是pe", "什么是pb"]
    if any(kw in question for kw in no_source_keywords) or not sources:
        # 检查是否是简单问候或纯概念问题
        if len(question) < 15 or not any(kw in question for kw in ["分析", "预测", "走势", "行情", "股价", "买入", "卖出", "推荐"]):
            return Send("summary", state)
    
    # ===== 根据 required_sources 构建路由列表 =====
    # 数据源节点映射
    source_node_map = {
        "index": "index_agent",
        "stock": "stock_agent", 
        "sentiment": "sentiment_agent",
        "theme": "theme_agent",
        "rag": "rag_agent",
        "search": "search_agent"
    }
    
    # 构建要执行的数据源节点列表
    nodes_to_execute = []
    
    for source in sources:
        node_name = source_node_map.get(source)
        if node_name:
            nodes_to_execute.append(Send(node_name, state))
    
    # 如果没有匹配的数据源，默认执行 summary
    if not nodes_to_execute:
        return Send("summary", state)
    
    return nodes_to_execute
