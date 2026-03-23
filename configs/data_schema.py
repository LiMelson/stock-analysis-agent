"""
核心数据模型

包含 RAG 系统的基础数据类：
- Document: 文档数据类
- SearchResult: 搜索结果数据类
- KnowledgeBase: 知识库数据类
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """文档数据类"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[str] = field(default_factory=list)
    embeddings: Optional[List[List[float]]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "metadata": self.metadata,
            "chunks_count": len(self.chunks),
            "created_at": self.created_at
        }


@dataclass
class SearchResult:
    """搜索结果数据类"""
    document_id: str
    chunk: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeBase:
    """知识库数据类"""
    id: str
    name: str
    description: str = ""
    documents: Dict[str, Document] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "documents_count": len(self.documents),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
