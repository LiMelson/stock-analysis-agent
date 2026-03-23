"""
RAG 检索器模块

RAGTool - 专注于检索逻辑。
知识库管理由 KnowledgeBaseManager 完成，通过参数传入。
"""

import warnings
from typing import List, Dict, Any, Optional

from configs.data_schema import SearchResult
from configs.model_config import LLMClient


class RAGTool:
    """
    RAG 工具类 - 专注于检索增强
    
    使用示例:
        kb = KnowledgeBaseManager(embedding_model)
        kb.init_knowledge_base()
        kb.add_document("doc.pdf")
        
        rag = RAGTool(kb)  # 使用已有的知识库
        results = rag.search("query", use_mqe=True)
    """
    
    def __init__(self, kb_manager, llm_client: Optional[LLMClient] = None):
        """
        初始化 RAG 工具
        
        Args:
            kb_manager: KnowledgeBaseManager 实例
            llm_client: LLMClient 实例（可选）
        """
        self.kb_manager = kb_manager
        self.llm_client = llm_client
        
        if self.llm_client is None:
            try:
                self.llm_client = LLMClient()
            except ValueError:
                self.llm_client = None
    
    # ==================== 检索增强组件 ====================
    
    def _expand_query(self, query: str, num_queries: int = 3) -> List[str]:
        """MQE: 扩展查询"""
        if not self.llm_client:
            return [query]
        
        expanded = [query]
        try:
            prompt = f"""原始查询: {query}

请生成 {num_queries} 个扩展查询。

示例:
原始查询: 量化投资
扩展查询:
1. 量化投资策略有哪些
2. 什么是量化交易方法
3. 量化投资基金的特点

原始查询: {query}
扩展查询:"""
            
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是一个专业的搜索查询扩展助手。",
                temperature=0.5
            )
            
            for line in response.strip().split('\n'):
                line = line.strip()
                if line:
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                            break
                    if line and line != query:
                        expanded.append(line)
        except Exception as e:
            warnings.warn(f"LLM 扩展失败: {e}")
        
        return expanded[:num_queries + 1]
    
    def _generate_hypothetical_docs(self, query: str, num_variants: int = 3) -> List[str]:
        """HyDE: 生成假设文档"""
        if not self.llm_client:
            return [query]
        
        docs = []
        angles = [
            "从定义和概念角度回答",
            "从实际应用和案例角度回答",
            "从优缺点和注意事项角度回答"
        ]
        
        for i in range(min(num_variants, len(angles))):
            prompt = f"""请基于你的知识，写一段关于以下问题的理想答案。
{angles[i]}

问题: {query}

理想答案（约100字）:"""
            try:
                doc = self.llm_client.generate(prompt=prompt, temperature=0.5, max_tokens=200)
                docs.append(doc)
            except Exception as e:
                warnings.warn(f"生成假设文档失败: {e}")
        
        return docs if docs else [query]
    
    def _hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 5,
        semantic_weight: float = 0.7
    ) -> List[SearchResult]:
        """混合检索（语义 + 关键词）"""
        
        embedding_model = self.kb_manager.embedding_model
        
        # 语义检索
        query_embedding = embedding_model.encode(query)
        semantic_results = self.kb_manager.search_vectors(query_embedding, top_k * 2)
        
        # 关键词检索
        all_keywords = [query] + (keywords or [])
        keyword_results = []
        
        # 安全获取文档，避免知识库未初始化时出错
        try:
            documents = self.kb_manager.documents
        except ValueError:
            documents = {}
        
        for doc_id, doc in documents.items():
            for chunk in doc.chunks:
                chunk_lower = chunk.lower()
                score = sum(1 for kw in all_keywords if kw.lower() in chunk_lower)
                if score > 0:
                    keyword_results.append(SearchResult(
                        document_id=doc_id,
                        chunk=chunk,
                        score=score,
                        metadata=doc.metadata
                    ))
        
        keyword_results.sort(key=lambda x: x.score, reverse=True)
        keyword_results = keyword_results[:top_k * 2]
        
        # 合并结果
        chunk_scores = {}
        
        for r in semantic_results:
            if r.chunk not in chunk_scores:
                chunk_scores[r.chunk] = {"semantic": r.score, "keyword": 0, "doc_id": r.document_id, "meta": r.metadata}
            else:
                chunk_scores[r.chunk]["semantic"] = max(chunk_scores[r.chunk]["semantic"], r.score)
        
        max_kw_score = max([r.score for r in keyword_results]) if keyword_results else 1
        for r in keyword_results:
            if r.chunk not in chunk_scores:
                chunk_scores[r.chunk] = {"semantic": 0, "keyword": r.score / max_kw_score, "doc_id": r.document_id, "meta": r.metadata}
            else:
                chunk_scores[r.chunk]["keyword"] = max(chunk_scores[r.chunk]["keyword"], r.score / max_kw_score)
        
        results = []
        for chunk, scores in chunk_scores.items():
            hybrid_score = semantic_weight * scores["semantic"] + (1 - semantic_weight) * scores["keyword"]
            results.append(SearchResult(
                document_id=scores["doc_id"],
                chunk=chunk,
                score=hybrid_score,
                metadata=scores["meta"]
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    # ==================== 对外接口 ====================
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_mqe: bool = False,
        use_hyde: bool = False
    ) -> Dict[str, Any]:
        """
        高级检索 - 组合 MQE + HyDE + 混合检索
        
        Returns:
            {"results": [...], "expanded_queries": [...], "hypothetical_docs": [...]}
        """
        all_queries = [query]
        expanded_queries = []
        hypothetical_docs = []
        
        if use_mqe and self.llm_client:
            expanded = self._expand_query(query)
            expanded_queries = expanded[1:]
            all_queries.extend(expanded_queries)
        
        if use_hyde and self.llm_client:
            hypothetical_docs = self._generate_hypothetical_docs(query)
            all_queries.extend(hypothetical_docs)
        
        all_results = []
        for q in all_queries:
            results = self._hybrid_search(q, top_k=top_k)
            all_results.extend(results)
        
        # 去重排序
        chunk_best = {}
        for r in all_results:
            if r.chunk not in chunk_best or r.score > chunk_best[r.chunk].score:
                chunk_best[r.chunk] = r
        
        unique_results = list(chunk_best.values())
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return {
            'results': unique_results[:top_k],
            'expanded_queries': expanded_queries,
            'hypothetical_docs': hypothetical_docs
        }
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_mqe: bool = False,
        use_hyde: bool = False,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        RAG 查询 - 返回格式化的检索结果
        
        Returns:
            {"context": "...", "chunks": [...], "sources": [...]}
        """
        search_result = self.search(
            query=question,
            top_k=top_k,
            use_mqe=use_mqe,
            use_hyde=use_hyde
        )
        
        results = search_result['results']
        
        if not results:
            return {"context": "", "chunks": [], "sources": []}
        
        context = "\n\n".join([f"[{i+1}] {r.chunk}" for i, r in enumerate(results)])
        
        return {
            "context": context,
            "chunks": [r.chunk for r in results],
            "sources": [
                {"document_id": r.document_id, "score": r.score, "metadata": r.metadata}
                for r in results
            ] if return_sources else []
        }
