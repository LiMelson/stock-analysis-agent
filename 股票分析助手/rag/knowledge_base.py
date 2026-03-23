"""
知识库管理模块

完全独立的知识库管理，不依赖 RAGTool。
负责：知识库初始化、文档管理、向量存储、保存加载。
"""

import os
import hashlib
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime

from configs.data_schema import Document, KnowledgeBase
from rag.document_processor import TextSplitter
from rag.extract import DocumentLoader
from rag.vector_store import RAGVectorStoreManager


class KnowledgeBaseManager:
    """
    知识库管理器 - 完全独立的文档和向量存储管理
    
    使用示例:
        kb = KnowledgeBaseManager(embedding_model)
        kb.init_knowledge_base("我的知识库")
        kb.add_document("document.pdf")
        kb.save("./vector_store")
    """
    
    def __init__(
        self,
        embedding_model,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.embedding_model = embedding_model
        self.text_splitter = TextSplitter(chunk_size, chunk_overlap)
        self.vector_store_manager = RAGVectorStoreManager(embedding_model)
        
        self.knowledge_base: Optional[KnowledgeBase] = None
        self.vector_store = None
        self._lock = threading.RLock()
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取嵌入向量"""
        return self.embedding_model.encode_batch(texts)
    
    def _generate_id(self, content: str) -> str:
        """生成文档ID"""
        return hashlib.md5(content[:1000].encode()).hexdigest()[:16]
    
    def _ensure_initialized(self):
        """确保知识库已初始化"""
        if self.knowledge_base is None:
            raise ValueError("知识库未初始化，请先调用 init_knowledge_base()")
    
    # ==================== 知识库生命周期 ====================
    
    def init_knowledge_base(
        self,
        name: str = "默认知识库",
        description: str = "",
        vector_store_dir: Optional[str] = None
    ) -> bool:
        """
        初始化知识库
        
        Args:
            name: 知识库名称
            description: 描述
            vector_store_dir: 向量存储目录（可选，用于加载已有存储）
            
        Returns:
            是否从已有存储加载
        """
        with self._lock:
            if self.knowledge_base is not None:
                print("[WARN] 知识库已存在")
                return False
            
            self.knowledge_base = KnowledgeBase(
                id="default_kb",
                name=name,
                description=description
            )
            
            # 加载或创建向量存储
            loaded = False
            if vector_store_dir:
                self.vector_store = self.vector_store_manager.load_store("default_kb", vector_store_dir)
                loaded = not self.vector_store.is_empty()
                if loaded:
                    print(f"[OK] 知识库已从 {vector_store_dir} 加载")
            
            if not loaded:
                self.vector_store = self.vector_store_manager.get_or_create_store("default_kb")
                print(f"[OK] 知识库已初始化: {name}")
            
            return loaded
    
    def clear(self) -> None:
        """清空知识库"""
        with self._lock:
            self._ensure_initialized()
            self.knowledge_base.documents.clear()
            self.knowledge_base.updated_at = datetime.now().isoformat()
            self.vector_store_manager.delete_store("default_kb")
            self.vector_store = self.vector_store_manager.get_or_create_store("default_kb")
            print("[OK] 知识库已清空")
    
    def save(self, vector_store_dir: str) -> bool:
        """保存向量存储"""
        if self.vector_store is None:
            print("[WARN] 向量存储未初始化")
            return False
        
        success = self.vector_store_manager.save_store("default_kb", vector_store_dir)
        if success:
            print(f"[OK] 已保存到: {vector_store_dir}")
        return success
    
    # ==================== 文档管理 ====================
    
    def add_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加文档到知识库"""
        with self._lock:
            self._ensure_initialized()
            
            content = DocumentLoader.load(file_path)
            doc_id = self._generate_id(file_path + str(os.path.getsize(file_path)))
            
            meta = (metadata or {}).copy()
            meta.update({
                "source": file_path,
                "filename": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "added_at": datetime.now().isoformat()
            })
            
            chunks = self.text_splitter.split(content)
            embeddings = self._get_embeddings(chunks)
            
            doc = Document(
                id=doc_id,
                content=content,
                metadata=meta,
                chunks=chunks,
                embeddings=embeddings
            )
            
            self.knowledge_base.documents[doc_id] = doc
            self.knowledge_base.updated_at = datetime.now().isoformat()
            
            for chunk, embedding in zip(chunks, embeddings):
                self.vector_store.add(doc_id, chunk, embedding, meta)
            
            print(f"[OK] 文档已添加: {meta['filename']}")
            return doc_id
    
    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加文本到知识库"""
        with self._lock:
            self._ensure_initialized()
            
            doc_id = self._generate_id(text)
            
            meta = (metadata or {}).copy()
            meta.update({
                "type": "text",
                "added_at": datetime.now().isoformat()
            })
            
            chunks = self.text_splitter.split(text)
            embeddings = self._get_embeddings(chunks)
            
            doc = Document(
                id=doc_id,
                content=text,
                metadata=meta,
                chunks=chunks,
                embeddings=embeddings
            )
            
            self.knowledge_base.documents[doc_id] = doc
            self.knowledge_base.updated_at = datetime.now().isoformat()
            
            for chunk, embedding in zip(chunks, embeddings):
                self.vector_store.add(doc_id, chunk, embedding, meta)
            
            print(f"[OK] 文本已添加")
            return doc_id
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档"""
        if self.knowledge_base is None:
            return []
        return [doc.to_dict() for doc in self.knowledge_base.documents.values()]
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        if self.knowledge_base is None:
            return {}
        
        return {
            **self.knowledge_base.to_dict(),
            "document_count": len(self.knowledge_base.documents),
            "documents": [doc.to_dict() for doc in self.knowledge_base.documents.values()]
        }
    
    # ==================== 检索接口（供 RAGTool 使用）====================
    
    def search_vectors(self, query_embedding: List[float], top_k: int = 10) -> List[Any]:
        """向量检索"""
        if self.vector_store is None:
            return []
        return self.vector_store.search(query_embedding, top_k)
    
    @property
    def documents(self) -> Dict[str, Document]:
        """获取所有文档"""
        self._ensure_initialized()
        return self.knowledge_base.documents
