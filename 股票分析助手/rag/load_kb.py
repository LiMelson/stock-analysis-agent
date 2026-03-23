"""
知识库批量加载脚本

将 docs 文件夹中的文档批量导入知识库，并持久化到 vector_stores 文件夹。
"""

import os
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.rag_retriever import RAGTool

#加国内镜像源
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
# 配置路径
DOCS_DIR = project_root / "docs"
VECTOR_STORE_DIR = project_root / "vector_stores"


def get_pdf_files(docs_dir: Path) -> List[Path]:
    """获取文档目录中的所有 PDF 文件"""
    if not docs_dir.exists():
        print(f"[X] 文档目录不存在: {docs_dir}")
        return []
    
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    print(f"[*] 发现 {len(pdf_files)} 个 PDF 文件")
    return pdf_files


def load_documents_to_kb(
    embedding_model: str = 'BAAI/bge-small-zh-v1.5',
    force_rebuild: bool = False
) -> RAGTool:
    """
    将 docs 文件夹中的文档加载到知识库
    
    Args:
        embedding_model: 嵌入模型名称
        force_rebuild: 是否强制重建知识库（忽略已有存储）
        
    Returns:
        初始化好的 RAGTool 实例
    """
    # 创建 RAG 工具
    print(f"[INFO] 初始化 RAG 工具 (模型: {embedding_model})...")
    rag = RAGTool(embedding_model=embedding_model)
    
    # 确保 vector_stores 目录存在
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 初始化知识库
    vector_store_path = str(VECTOR_STORE_DIR)
    
    if not force_rebuild:
        # 尝试加载已有知识库
        loaded = rag.init_knowledge_base(
            name="股市操练大全知识库",
            description="包含股市操练大全系列书籍的股市投资知识",
            vector_store_dir=vector_store_path
        )
        if loaded:
            print(f"[INFO] 知识库信息: {rag.get_knowledge_base_info()}")
            return rag
    else:
        # 强制重建
        rag.init_knowledge_base(
            name="股市操练大全知识库",
            description="包含股市操练大全系列书籍的股市投资知识"
        )
        print("[INFO] 强制重建知识库...")
    
    # 获取 PDF 文件列表
    pdf_files = get_pdf_files(DOCS_DIR)
    if not pdf_files:
        print("[WARN] 没有找到 PDF 文件")
        return rag
    
    # 批量添加文档
    print(f"\n[INFO] 开始导入文档...")
    success_count = 0
    failed_files = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] 正在处理: {pdf_file.name}")
        try:
            doc_id = rag.add_document(
                file_path=str(pdf_file),
                metadata={
                    "category": "股市操练大全",
                    "book_name": pdf_file.stem
                }
            )
            success_count += 1
            print(f"   [OK] 成功导入 (ID: {doc_id[:8]}...)")
        except Exception as e:
            print(f"   [FAIL] 导入失败: {e}")
            failed_files.append((pdf_file.name, str(e)))
    
    # 保存知识库
    print(f"\n[INFO] 保存知识库到: {vector_store_path}")
    rag.save_knowledge_base(vector_store_path)
    
    # 打印统计
    print(f"\n[INFO] 导入统计:")
    print(f"   - 成功: {success_count}/{len(pdf_files)}")
    print(f"   - 失败: {len(failed_files)}")
    
    if failed_files:
        print(f"\n[FAIL] 失败的文件:")
        for name, error in failed_files:
            print(f"   - {name}: {error}")
    
    # 打印知识库信息
    kb_info = rag.get_knowledge_base_info()
    print(f"\n[INFO] 知识库信息:")
    print(f"   - 名称: {kb_info.get('name')}")
    print(f"   - 文档数: {kb_info.get('document_count', 0)}")
    
    return rag


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量导入文档到知识库")
    parser.add_argument(
        "--rebuild", 
        action="store_true", 
        help="强制重建知识库（删除已有数据）"
    )
    parser.add_argument(
        "--model", 
        default="BAAI/bge-small-zh-v1.5",
        help="嵌入模型名称（默认: BAAI/bge-small-zh-v1.5）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("[INFO] 股市操练大全知识库批量导入工具")
    print("=" * 60)
    
    # 检查依赖
    try:
        import faiss
    except ImportError:
        print("\n[FAIL] 请先安装 faiss-cpu: pip install faiss-cpu")
        sys.exit(1)
    
    try:
        from markitdown import MarkItDown
    except ImportError:
        print("\n[FAIL] 请先安装 markitdown: pip install markitdown")
        sys.exit(1)
    
    # 执行导入
    rag = load_documents_to_kb(
        embedding_model=args.model,
        force_rebuild=args.rebuild
    )
    
    print("\n" + "=" * 60)
    print("[DONE] 知识库导入完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
