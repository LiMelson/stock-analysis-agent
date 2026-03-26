"""
指数数据源模块 - 基于 Tushare Pro

提供大盘指数相关数据
注册地址：https://tushare.pro/register
"""

import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.state import AgentState
from configs.cache import cached


class IndexDataSource:
    """
    大盘指数数据源 - 基于 Tushare Pro
    """
    
    INDICES = {
        "000001": {"name": "上证指数", "ts_code": "000001.SH"},
        "399001": {"name": "深证成指", "ts_code": "399001.SZ"},
        "399006": {"name": "创业板指", "ts_code": "399006.SZ"},
        "000300": {"name": "沪深300", "ts_code": "000300.SH"},
        "000688": {"name": "科创50", "ts_code": "000688.SH"},
    }
    
    def __init__(self):
        self.name = "index"
        self.description = "大盘指数数据（Tushare Pro）"
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
    
    @cached(ttl=300, key_prefix="index")
    def fetch(self, index_code: Optional[str] = None) -> Dict[str, Any]:
        """获取大盘指数数据"""
        try:
            indices_data = []
            
            for code, info in self.INDICES.items():
                try:
                    # 使用 Tushare Pro 获取指数日线
                    df = self.pro.index_daily(
                        ts_code=info["ts_code"],
                        start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
                        end_date=datetime.now().strftime("%Y%m%d")
                    )
                    
                    if df is None or df.empty:
                        continue
                    
                    latest = df.iloc[0]  # Tushare 返回倒序
                    
                    indices_data.append({
                        "code": code,
                        "name": info["name"],
                        "date": latest["trade_date"],
                        "close": float(latest["close"]),
                        "open": float(latest["open"]),
                        "high": float(latest["high"]),
                        "low": float(latest["low"]),
                        "change": float(latest.get("pct_chg", 0)),
                        "change_amount": float(latest.get("change", 0)),
                        "volume": float(latest.get("vol", 0)),
                    })
                    
                except Exception as e:
                    print(f"  [WARN] 获取指数 {code} 失败: {e}")
                    continue
            
            if not indices_data:
                raise Exception("无法获取任何指数数据，请检查 Tushare Token 是否有效且有权限")
            
            return {
                "status": "success",
                "indices": indices_data,
                "evaluation": self._generate_evaluation(indices_data)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "indices": [],
                "evaluation": f"获取指数数据失败: {e}"
            }
    
    def _generate_evaluation(self, indices_data: List[Dict]) -> str:
        """生成大盘指数评价文本"""
        lines = ["## 大盘指数行情\n"]
        
        priority = {"上证指数": 1, "深证成指": 2, "创业板指": 3, "沪深300": 4, "科创50": 5}
        indices_data.sort(key=lambda x: priority.get(x["name"], 99))
        
        for idx in indices_data:
            emoji = "📈" if idx["change"] > 0 else "📉" if idx["change"] < 0 else "➡️"
            lines.append(f"### {idx['name']} ({idx['code']})")
            lines.append(f"- **收盘**: {idx['close']:.2f} {emoji}")
            lines.append(f"- **涨跌**: {idx['change']:+.2f}%")
            lines.append("")
        
        # 市场综述
        sh = next((x for x in indices_data if x["code"] == "000001"), None)
        cy = next((x for x in indices_data if x["code"] == "399006"), None)
        
        if sh and cy:
            lines.append("### 市场综述")
            if sh["change"] > 0 and cy["change"] > 0:
                lines.append("- 大盘整体上涨，市场情绪积极")
            elif sh["change"] < 0 and cy["change"] < 0:
                lines.append("- 大盘整体下跌，需注意风险")
            else:
                lines.append("- 市场分化，结构性行情")
        
        return "\n".join(lines)
    
    @classmethod
    def as_node(cls):
        """创建 LangGraph 节点函数"""
        agent = cls()
        
        def index_node(state: AgentState) -> Dict[str, Any]:
            print("    [IndexAgent] 正在获取大盘指数...", flush=True)
            result = agent.fetch()
            
            if result["status"] == "success":
                print(f"    [IndexAgent] 获取成功: {len(result.get('indices', []))} 个指数", flush=True)
                return {
                    "index_result": result["evaluation"],
                    "index_sources": [{"source": "tushare", "count": len(result['indices'])}]
                }
            else:
                print(f"    [IndexAgent] 获取失败: {result.get('error', '未知错误')}", flush=True)
                return {
                    "index_result": f"获取指数数据失败: {result.get('error', '未知错误')}",
                    "index_sources": []
                }
        
        return index_node
