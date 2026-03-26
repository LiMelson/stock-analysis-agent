"""
Intent & Requirement Agent 模块

职责：理解用户意图和需求
- 意图分类：判断用户想要什么
- 需求提取：提取标的、关注点、隐含需求等

输出：intent_type + requirement_analysis
"""

from typing import Dict, Any, Optional, List, Literal
from configs.model_config import LLMClient
from ..base_agent import BaseAgent, agent_run_wrapper, AgentTimer

# 意图类型定义
IntentType = Literal[
    "real_time_quote",      # 实时行情
    "historical_report",    # 历史数据
    "investment_analysis",  # 投资分析
    "general_question",     # 一般问题
    "greeting"              # 问候
]


class IntentRequirementAgent(BaseAgent):
    """
    意图与需求分析 Agent
    
    只负责理解用户问题，不涉及数据源选择
    """
    
    def __init__(self):
        super().__init__("IntentRequirementAgent")
        self.llm_client = LLMClient()
    
    @agent_run_wrapper
    def analyze(self, question: str, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        分析用户意图和需求
        """
        self._log_step("意图与需求分析", f"question={question}")
        
        system_prompt = """你是一位专业的意图分析和需求理解专家。

你的任务是理解用户问题，输出意图类型和需求细节。

## 意图类型定义

1. **real_time_quote** (实时行情)
   - 特征：询问最新股价、今天行情、实时数据、当前价格等时效性信息
   - 示例: "茅台现在多少钱？", "今天大盘怎么样？"

2. **historical_report** (历史数据)
   - 特征：询问历史财报、历史业绩、过往数据等
   - 示例: "茅台2023年财报", "比亚迪的ROE是多少？"

3. **investment_analysis** (投资分析)
   - 特征：需要投资建议、分析预测、买卖建议、估值分析等
   - 示例: "分析一下茅台", "现在能买比亚迪吗？"

4. **general_question** (一般问题)
   - 特征：关于股票概念、投资理论、基础知识等的询问
   - 示例: "什么是市盈率？", "K线图怎么看？"

5. **greeting** (问候)
   - 特征：打招呼、寒暄等
   - 示例: "你好", "在吗"

## 需求理解维度

1. **target_stock**: 涉及的股票/基金代码或公司名称（如有）
2. **focus_areas**: 用户最关心的维度列表（如：价格、财报、走势、风险等）
3. **implicit_needs**: 用户可能未明确表达但需要的分析内容
4. **question_type**: 个股分析/行业分析/大盘走势/投资策略/概念解释/其他
5. **urgency_level**: high/medium/low

请输出JSON格式：
{
    "intent_type": "real_time_quote|historical_report|investment_analysis|general_question|greeting",
    "intent_confidence": 0.0-1.0,
    "intent_reasoning": "意图分类理由",
    "target_stock": "股票/基金代码或公司名称，没有则填null",
    "focus_areas": ["关注维度1", "关注维度2"],
    "implicit_needs": "隐含需求描述",
    "question_type": "个股分析|行业分析|大盘走势|投资策略|概念解释|其他",
    "urgency_level": "high|medium|low"
}"""

        history_text = ""
        if chat_history:
            history_text = "\n\n历史对话记录（最近3轮）：\n"
            for msg in chat_history[-3:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
        
        prompt = f"""请对用户问题进行意图分类和需求分析。

用户问题：{question}{history_text}

请判断意图类型并提取需求信息，只输出JSON格式："""

        with AgentTimer("意图与需求分析", self.logger):
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
        
        # 解析结果
        result = self._parse_response(response, question)
        
        self.logger.info(f"意图分析: {result['intent_type']}, 标的: {result.get('target_stock', '无')}")
        
        return result
    
    def _parse_response(self, response: str, question: str) -> Dict[str, Any]:
        """解析 LLM 响应"""
        import json
        import re
        
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            # 确保必要字段存在
            result.setdefault("intent_type", "unknown")
            result.setdefault("intent_confidence", 0.5)
            result.setdefault("intent_reasoning", "")
            result.setdefault("target_stock", None)
            result.setdefault("focus_areas", [])
            result.setdefault("implicit_needs", "需要全面分析")
            result.setdefault("question_type", "其他")
            result.setdefault("urgency_level", "medium")
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"解析失败: {e}，返回默认结果")
            result = {
                "intent_type": "unknown",
                "intent_confidence": 0.0,
                "intent_reasoning": f"解析失败: {str(e)}",
                "target_stock": None,
                "focus_areas": [],
                "implicit_needs": "解析失败，需要重新分析",
                "question_type": "其他",
                "urgency_level": "medium"
            }
        
        return result
