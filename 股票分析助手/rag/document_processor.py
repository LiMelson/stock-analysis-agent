"""
文档处理模块

提供 Markdown 感知文本分割器。
"""

from typing import List, Dict

# 尝试导入可选依赖
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None


class TextSplitter:
    """
    Markdown 感知文本分割器
    
    分块策略：
    1. 将文本转换为 Markdown 格式
    2. 按标题层级分割
    3. 在标题块内按 token 数量细分割（chunk_size=1000, overlap=200）
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, encoding_name: str = "cl100k_base"):
        """
        Args:
            chunk_size: 每个块的目标 token 数
            chunk_overlap: 块之间的重叠 token 数
            encoding_name: tiktoken 编码名称
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self._encoding = None
        
        if HAS_TIKTOKEN:
            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                pass
    
    def _get_token_count(self, text: str) -> int:
        """获取文本的 token 数量"""
        if self._encoding:
            return len(self._encoding.encode(text))
        # 降级方案：按字符估算（中文约 1.5 字符/token，英文约 4 字符/token）
        return len(text) // 2
    
    def _text_to_markdown(self, text: str) -> str:
        """
        将纯文本转换为 Markdown 格式
        
        启发式规则：
        - 短行且全大写 → 标题
        - 以数字/符号开头的短行 → 小标题
        - 连续空行 → 段落分隔
        """
        import re
        
        lines = text.split('\n')
        md_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                md_lines.append('')
                continue
            
            # 检测标题模式
            # 模式1：全大写短行（可能是章节标题）
            if len(stripped) < 50 and stripped.isupper() and any(c.isalpha() for c in stripped):
                md_lines.append(f"## {stripped}")
                continue
            
            # 模式2：数字编号 + 文本（如 "1. 引言" 或 "第1章 标题"）
            if re.match(r'^(\d+[\.、]|第[一二三四五六七八九十\d]+章|Chapter \d+)', stripped) and len(stripped) < 100:
                md_lines.append(f"## {stripped}")
                continue
            
            # 模式3：特殊符号包裹（如 **标题** 或 ==标题==）
            if re.match(r'^[\*\=\#\-]{2,}[^\*\=\#\-]+[\*\=\#\-]{2,}$', stripped):
                clean = stripped.strip('*=#- ')
                md_lines.append(f"## {clean}")
                continue
            
            # 模式4：短行以冒号结尾（可能是小标题）
            if len(stripped) < 40 and stripped.endswith('：'):
                md_lines.append(f"### {stripped[:-1]}")
                continue
            
            md_lines.append(stripped)
        
        return '\n'.join(md_lines)
    
    def _split_by_headers(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        按 Markdown 标题分割文本
        
        Returns:
            列表，每个元素包含标题层级和对应内容
        """
        import re
        
        # 匹配 Markdown 标题（# ## ###）
        header_pattern = re.compile(r'^(#{1,6})\s*(.+)$', re.MULTILINE)
        
        sections = []
        current_title = ""
        current_level = 0
        current_content = []
        
        lines = markdown_text.split('\n')
        
        for line in lines:
            match = header_pattern.match(line)
            if match:
                # 保存上一个区块
                if current_content:
                    sections.append({
                        'title': current_title,
                        'level': current_level,
                        'content': '\n'.join(current_content).strip()
                    })
                
                # 开始新区块
                hashes, title = match.groups()
                current_level = len(hashes)
                current_title = title.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # 保存最后一个区块
        if current_content or current_title:
            sections.append({
                'title': current_title,
                'level': current_level,
                'content': '\n'.join(current_content).strip()
            })
        
        # 如果没有检测到标题，将整个文本作为一个区块
        if not sections:
            sections.append({
                'title': "",
                'level': 0,
                'content': markdown_text.strip()
            })
        
        return sections
    
    def _split_by_tokens(self, text: str) -> List[str]:
        """
        按 token 数量分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        # 如果文本本身小于 chunk_size，直接返回
        if self._get_token_count(text) <= self.chunk_size:
            return [text]
        
        # 按句子分割
        import re
        # 支持中英文句子分隔符
        sentence_pattern = re.compile(r'([。！？.!?\n]+)')
        sentences = sentence_pattern.split(text)
        
        # 合并分隔符和句子
        fragments = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                fragments.append(sentences[i] + sentences[i + 1])
            else:
                fragments.append(sentences[i])
        if len(sentences) % 2 == 1:
            fragments.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for fragment in fragments:
            fragment_tokens = self._get_token_count(fragment)
            
            # 如果单个片段超过 chunk_size，需要进一步分割
            if fragment_tokens > self.chunk_size:
                # 先保存当前累积的内容
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                # 按字符分割，支持中英文混合
                sub_chunks = self._split_long_text(fragment)
                chunks.extend(sub_chunks[:-1])  # 除最后一个都加入结果
                
                # 最后一个作为当前块继续累积
                if sub_chunks:
                    current_chunk = sub_chunks[-1]
                    current_tokens = self._get_token_count(current_chunk)
            
            # 正常情况：累加片段
            elif current_tokens + fragment_tokens > self.chunk_size:
                chunks.append(current_chunk.strip())
                
                # 计算重叠部分
                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + fragment
                    current_tokens = self._get_token_count(current_chunk)
                else:
                    current_chunk = fragment
                    current_tokens = fragment_tokens
            else:
                current_chunk += fragment
                current_tokens += fragment_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _split_long_text(self, text: str) -> List[str]:
        """
        分割超长文本（单个句子超过 chunk_size）
        
        使用字符级分割，支持中英文混合
        """
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # 按字符迭代（正确处理中英文）
        i = 0
        while i < len(text):
            # 尝试取一个"词"（中文单字，或连续的英文字母/数字）
            if i < len(text) and ord(text[i]) > 127:  # 中文字符
                char = text[i]
                i += 1
            else:
                # 连续的 ASCII 字符作为一个词
                j = i
                while j < len(text) and ord(text[j]) <= 127 and text[j] not in ' \t\n':
                    j += 1
                if j == i:
                    char = text[i]
                    i += 1
                else:
                    char = text[i:j]
                    i = j
            
            char_tokens = self._get_token_count(char)
            
            if current_tokens + char_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # 保留重叠部分
                    if self.chunk_overlap > 0:
                        current_chunk = self._get_overlap_text(current_chunk) + char
                        current_tokens = self._get_token_count(current_chunk)
                    else:
                        current_chunk = char
                        current_tokens = char_tokens
                else:
                    # 单个字符就超过限制，直接添加
                    chunks.append(char)
                    current_chunk = ""
                    current_tokens = 0
            else:
                current_chunk += char
                current_tokens += char_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _get_overlap_text(self, text: str) -> str:
        """
        从文本末尾获取重叠部分
        
        从后向前取字符，直到达到 chunk_overlap 的 token 数量
        """
        if not text or self.chunk_overlap <= 0:
            return ""
        
        overlap_tokens = 0
        overlap_chars = []
        
        # 从后向前遍历字符
        for char in reversed(text):
            char_tokens = self._get_token_count(char)
            if overlap_tokens + char_tokens <= self.chunk_overlap:
                overlap_chars.append(char)
                overlap_tokens += char_tokens
            else:
                break
        
        return ''.join(reversed(overlap_chars))
    
    def split(self, text: str) -> List[str]:
        """
        执行完整的分块流程
        
        1. 转换为 Markdown
        2. 按标题分割
        3. 每个标题块内按 token 细分割
        
        Args:
            text: 原始文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        # 步骤1：转换为 Markdown
        markdown_text = self._text_to_markdown(text)
        
        # 步骤2：按标题分割
        sections = self._split_by_headers(markdown_text)
        
        # 步骤3：每个区块按 token 分割
        all_chunks = []
        for section in sections:
            header = section['title']
            content = section['content']
            
            if not content and not header:
                continue
            
            # 组合标题和内容
            if header:
                full_text = f"## {header}\n\n{content}"
            else:
                full_text = content
            
            # 如果整个区块不大，直接保留
            if self._get_token_count(full_text) <= self.chunk_size:
                all_chunks.append(full_text)
            else:
                # 需要在区块内进一步分割
                sub_chunks = self._split_by_tokens(content)
                for i, sub_chunk in enumerate(sub_chunks):
                    if i == 0 and header:
                        # 第一个块保留标题
                        all_chunks.append(f"## {header}\n\n{sub_chunk}")
                    else:
                        # 后续块添加标题引用
                        if header:
                            all_chunks.append(f"## {header} (续)\n\n{sub_chunk}")
                        else:
                            all_chunks.append(sub_chunk)
        
        return all_chunks if all_chunks else [text]
