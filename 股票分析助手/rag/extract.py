"""
文档提取模块

提供多格式文档加载器，使用 MarkItDown 作为统一转换引擎。
"""

import warnings
from typing import Optional
from pathlib import Path

# 尝试导入可选依赖
try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False
    MarkItDown = None


class DocumentLoader:
    """多格式文档加载器 - 使用 MarkItDown 作为统一转换引擎"""
    
    _md_instance: Optional[MarkItDown] = None
    
    @classmethod
    def _get_markitdown(cls) -> MarkItDown:
        """获取 MarkItDown 实例（单例模式）"""
        if not HAS_MARKITDOWN:
            raise ImportError(
                "请安装 MarkItDown: pip install markitdown\n"
                "MarkItDown 支持 PDF、Word、Excel、PPT、图片等多种格式"
            )
        
        if cls._md_instance is None:
            cls._md_instance = MarkItDown()
        return cls._md_instance
    
    # 明确定义文本格式和二进制格式
    TEXT_FORMATS = {'.txt', '.md', '.markdown', '.py', '.js', '.html', '.css', 
                    '.json', '.csv', '.xml', '.yaml', '.yml', '.ini', '.conf',
                    '.log', '.sh', '.bat', '.java', '.c', '.cpp', '.h', '.go',
                    '.rs', '.rb', '.php', '.ts', '.tsx', '.jsx', '.vue'}
    
    BINARY_FORMATS = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt',
                      '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp',
                      '.zip'}
    
    @classmethod
    def load(cls, file_path: str) -> str:
        """
        加载文档内容，使用 MarkItDown 统一转换为 Markdown
        
        Args:
            file_path: 文件路径
            
        Returns:
            Markdown 格式的文档内容
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = path.suffix.lower()
        
        # 纯文本文件直接读取
        if suffix in cls.TEXT_FORMATS:
            return cls._load_text(file_path)
        
        # 二进制格式必须使用 MarkItDown，失败时不降级到文本读取
        if suffix in cls.BINARY_FORMATS:
            try:
                return cls._convert_with_markitdown(file_path)
            except Exception as e:
                raise RuntimeError(
                    f"无法转换二进制文件 {file_path}: {e}\n"
                    f"请确保已安装 MarkItDown: pip install markitdown"
                ) from e
        
        # 其他格式尝试使用 MarkItDown 转换
        try:
            return cls._convert_with_markitdown(file_path)
        except Exception as e:
            # 未知格式，尝试作为文本读取（仅当不是明显二进制文件时）
            warnings.warn(f"MarkItDown 转换失败: {e}，尝试作为文本读取")
            return cls._load_text(file_path)
    
    @staticmethod
    def _load_text(file_path: str) -> str:
        """加载文本文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"无法解码文件: {file_path}")
    
    @classmethod
    def _convert_with_markitdown(cls, file_path: str) -> str:
        """
        使用 MarkItDown 将文档转换为 Markdown
        
        Args:
            file_path: 文件路径
            
        Returns:
            Markdown 格式的文本内容
        """
        md = cls._get_markitdown()
        result = md.convert(file_path)
        return result.text_content if hasattr(result, 'text_content') else str(result)
