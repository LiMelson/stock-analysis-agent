"""
市场情绪数据源模块 - 基于 Tushare Pro

通过涨跌停家数统计反映市场情绪
注册地址：https://tushare.pro/register
"""

from typing import Dict, Any
from datetime import datetime
from core.state import AgentState
from configs.cache import cached


class SentimentDataSource:
    """
    市场情绪数据源 - 基于 Tushare Pro
    """
    
    def __init__(self):
        self.name = "sentiment"
        self.description = "市场情绪数据（Tushare Pro）"
        self.pro = self._init_tushare()
    
    def _init_tushare(self):
        """初始化 Tushare Pro"""
        import tushare as ts
        import os
        
        token = os.getenv("TUSHARE_TOKEN")
        if not token or token == "your-tushare-token-here":
            raise ValueError("请先在 .env 文件中设置正确的 TUSHARE_TOKEN")
        
        ts.set_token(token)
        return ts.pro_api()
    
    @cached(ttl=300, key_prefix="sentiment")
    def fetch(self) -> Dict[str, Any]:
        """获取市场情绪数据"""
        try:
            # 使用 Tushare Pro 获取涨跌停统计
            trade_date = datetime.now().strftime("%Y%m%d")
            
            try:
                # 涨停列表
                df_zt = self.pro.limit_list(trade_date=trade_date, limit_type='U')
                up_count = len(df_zt) if df_zt is not None else 0
            except:
                up_count = 0
            
            try:
                # 跌停列表
                df_dt = self.pro.limit_list(trade_date=trade_date, limit_type='D')
                down_count = len(df_dt) if df_dt is not None else 0
            except:
                down_count = 0
            
            # 计算情绪得分
            score = up_count - down_count
            
            # 确定情绪等级
            if score < -50:
                level, emoji = "极度恐慌", "😱"
                desc = "跌停家数远多于涨停，市场恐慌"
            elif score < 0:
                level, emoji = "偏悲观", "😰"
                desc = "跌多涨少，市场情绪谨慎"
            elif score < 50:
                level, emoji = "中性", "😐"
                desc = "涨跌平衡，情绪中性"
            elif score < 100:
                level, emoji = "偏乐观", "🙂"
                desc = "涨多跌少，情绪积极"
            else:
                level, emoji = "极度贪婪", "🚀"
                desc = "涨停潮，市场过热"
            
            evaluation = f"""## 市场情绪分析

### 涨跌停统计
- **涨停家数**: {up_count}
- **跌停家数**: {down_count}
- **情绪得分**: {score:+.0f} {emoji}
- **情绪等级**: {level}

### 市场评价
{desc}
"""
            
            return {
                "status": "success",
                "up_count": up_count,
                "down_count": down_count,
                "score": score,
                "level": level,
                "evaluation": evaluation
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "evaluation": f"获取市场情绪失败: {e}"
            }
    
    @classmethod
    def as_node(cls):
        """创建 LangGraph 节点函数"""
        agent = cls()
        
        def sentiment_node(state: AgentState) -> Dict[str, Any]:
            print("    [SentimentAgent] 正在获取市场情绪...", flush=True)
            result = agent.fetch()
            
            if result["status"] == "success":
                print(f"    [SentimentAgent] 获取成功: 涨停{result.get('up_count', 0)}, 跌停{result.get('down_count', 0)}", flush=True)
                return {
                    "sentiment_result": result["evaluation"],
                    "sentiment_sources": [{"source": "tushare"}]
                }
            else:
                print(f"    [SentimentAgent] 获取失败: {result.get('error', '未知错误')}", flush=True)
                return {
                    "sentiment_result": f"获取市场情绪失败: {result.get('error', '未知错误')}",
                    "sentiment_sources": []
                }
        
        return sentiment_node
