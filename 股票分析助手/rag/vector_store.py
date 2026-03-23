"""
向量存储模块

提供基于 FAISS 的向量存储实现，支持持久化到磁盘。
"""
import pickle
from typing import List, Dict, Any
from pathlib import Path

# 可选依赖
try:
    import numpy as np
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    np = None
    faiss = None

from configs.data_schema import SearchResult


class FaissVectorStore:
    """纯内存版 FAISS 向量存储"""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension          # 向量维度
        self.doc_ids: List[str] = []        # 存储文档ID
        self.chunks: List[str] = []         # 存储文本片段
        self.metadata: List[Dict] = []      # 存储元数据
        self.index = None                    # FAISS索引对象
        self._id_map: Dict[int, int] = {}    # FAISS内部ID到列表索引的映射
        self._next_id = 0                     # 下一个可用的FAISS内部ID

    def add(self, doc_id: str, chunk: str, embedding: List[float], metadata: Dict[str, Any]):
        if not HAS_FAISS:
            raise ImportError("请安装：pip install faiss-cpu")
    
        # 转换并归一化向量
        emb = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(emb)  # L2归一化，使向量长度为1
    
        # 如果是第一个向量，创建内积索引
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
    
        # 添加到FAISS索引
        self.index.add(emb)
    
        # 记录映射关系
        self._id_map[self._next_id] = len(self.doc_ids)
        self._next_id += 1
    
        # 存储原始数据
        self.doc_ids.append(doc_id)
        self.chunks.append(chunk)
        self.metadata.append(metadata)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        if not HAS_FAISS or self.index is None or self.index.ntotal == 0:
            return []

        q = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            arr_idx = self._id_map.get(int(idx), int(idx))
            if arr_idx < len(self.doc_ids):
                results.append(SearchResult(
                    document_id=self.doc_ids[arr_idx],
                    chunk=self.chunks[arr_idx],
                    score=float(score),
                    metadata=self.metadata[arr_idx]
                ))
        return results

    def clear(self):
        self.doc_ids.clear()
        self.chunks.clear()
        self.metadata.clear()
        self.index = None
        self._id_map.clear()
        self._next_id = 0
    
    def save(self, save_dir: str) -> None:
        """
        保存向量存储到目录

        Args:
            save_dir: 保存目录路径
        """
        import os
        
        if not HAS_FAISS:
            raise ImportError("请安装：pip install faiss-cpu")

        # 转换为绝对路径并创建目录
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # 保存 FAISS 索引（序列化到内存再写入文件）
        if self.index is not None:
            index_path = os.path.join(save_dir, "faiss.index")
            # 使用 serialize 方法避免文件句柄问题
            serialized = faiss.serialize_index(self.index)
            with open(index_path, "wb") as f:
                import numpy as np
                np.save(f, serialized)

        # 保存元数据（使用 pickle）
        metadata = {
            "dimension": self.dimension,
            "doc_ids": self.doc_ids,
            "chunks": self.chunks,
            "metadata": self.metadata,
            "_id_map": self._id_map,
            "_next_id": self._next_id
        }
        meta_path = os.path.join(save_dir, "metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
    
    def load(self, save_dir: str) -> bool:
        """
        从目录加载向量存储
        
        Args:
            save_dir: 保存目录路径
            
        Returns:
            是否成功加载
        """
        import os
        
        if not HAS_FAISS:
            raise ImportError("请安装：pip install faiss-cpu")
        
        index_file = os.path.join(save_dir, "faiss.index")
        metadata_file = os.path.join(save_dir, "metadata.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            return False
        
        try:
            # 加载元数据
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
            
            self.dimension = data["dimension"]
            self.doc_ids = data["doc_ids"]
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            self._id_map = data["_id_map"]
            self._next_id = data["_next_id"]
            
            # 加载 FAISS 索引（从内存反序列化）
            with open(index_file, "rb") as f:
                import numpy as np
                serialized = np.load(f, allow_pickle=True)
            self.index = faiss.deserialize_index(serialized)
            
            return True
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            self.clear()
            return False
    
    def is_empty(self) -> bool:
        """检查向量存储是否为空"""
        return self.index is None or self.index.ntotal == 0


class RAGVectorStoreManager:
    """向量存储管理器"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self._vector_stores: Dict[str, FaissVectorStore] = {}

    def get_or_create_store(self, kb_id: str) -> FaissVectorStore:
        if kb_id not in self._vector_stores:
            dim = self.embedding_model.get_dimension()
            self._vector_stores[kb_id] = FaissVectorStore(dimension=dim)
        return self._vector_stores[kb_id]

    def get_store(self, kb_id: str) -> FaissVectorStore:
        return self._vector_stores.get(kb_id)

    def delete_store(self, kb_id: str):
        self._vector_stores.pop(kb_id, None)
    
    def save_store(self, kb_id: str, save_dir: str) -> bool:
        """
        保存指定知识库的向量库到磁盘
        """
        import os
        
        if kb_id not in self._vector_stores:
            return False
    
        kb_save_dir = os.path.join(save_dir, kb_id)
        
        # 确保目录存在
        os.makedirs(kb_save_dir, exist_ok=True)
        
        store = self._vector_stores[kb_id]
        store.save(kb_save_dir)
        return True
    
    def load_store(self, kb_id: str, save_dir: str) -> FaissVectorStore:
        """
        加载指定知识库的向量存储
        
        Args:
            kb_id: 知识库 ID
            save_dir: 保存目录
            
        Returns:
            加载的向量存储实例
        """
        # 使用 os.path.join 确保路径正确拼接
        import os
        kb_save_dir = os.path.join(save_dir, kb_id)
        
        # 创建新的存储实例
        dim = self.embedding_model.get_dimension()
        store = FaissVectorStore(dimension=dim)
        
        # 尝试加载
        if os.path.exists(kb_save_dir) and store.load(kb_save_dir):
            self._vector_stores[kb_id] = store
            return store
        
        # 加载失败，返回空存储
        return store
