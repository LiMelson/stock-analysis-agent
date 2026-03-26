"""
RAG 数据源模块 - 知识库检索
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.rag_retriever import RAGTool


class RAGDataSource:
    """
    RAG 数据源 - 基于知识库的检索
    
    直接返回检索到的内容，不做 LLM 总结
    """
    
    def __init__(self, rag_tool: 'RAGTool'):
        self.name = "rag"
        self.description = "知识库检索（个股基本面、投资理论、行业知识）"
        self.rag_tool = rag_tool
    
    def fetch(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        从知识库检索信息
        """
        try:
            result = self.rag_tool.query(
                question=query,
                top_k=top_k,
                return_sources=True
            )
            
            context = result.get("context", "")
            sources = result.get("sources", [])
            
            if not context:
                return {
                    "status": "no_results",
                    "content": "未在知识库中找到相关信息",
                    "sources": []
                }
            
            return {
                "status": "success",
                "content": context,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "content": f"检索失败: {str(e)}",
                "sources": []
            }
    
    @classmethod
    def as_node(cls, rag_tool: 'RAGTool'):
        """
        创建 LangGraph 节点函数
        
        Args:
            rag_tool: RAGTool 实例
            
        Returns:
            LangGraph 节点函数
        """
        agent = cls(rag_tool)
        
        def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            RAG 检索节点
            """
            from core.state import AgentState
            
            # 优先使用 PlanAgent 生成的 rag_query
            query = state.get("rag_query", "")
            if not query:
                query = state.get("question", "")
            
            if not query:
                return {
                    "rag_result": "未提供查询内容",
                    "rag_sources": []
                }
            
            # 执行检索
            result = agent.fetch(query)
            
            return {
                "rag_result": result["content"],
                "rag_sources": result["sources"]
            }
        
        return rag_node
