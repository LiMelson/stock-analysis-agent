"""
大盘指数数据源 - 基于 Tushare Pro

Tushare Pro 优点：
- 官方 API，稳定可靠，不反爬
- 数据质量高，更新及时
- 免费版额度足够个人使用

使用方式：
1. 注册获取 token：https://tushare.pro/register
2. 在 .env 文件中设置 TUSHARE_TOKEN
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.state import AgentState
from configs.cache import cached


class IndexDataSource:
    """
    大盘指数数据源 - 基于 Tushare Pro
    """
    
    # 指数代码映射（Tushare格式）
    INDICES = {
        "000001.SH": {"name": "上证指数", "code": "000001"},
        "399001.SZ": {"name": "深证成指", "code": "399001"},
        "399006.SZ": {"name": "创业板指", "code": "399006"},
        "000300.SH": {"name": "沪深300", "code": "000300"},
        "000688.SH": {"name": "科创50", "code": "000688"},
    }
    
    def __init__(self):
        self.name = "index"
        self.description = "大盘指数数据（上证指数、深证成指、创业板指等）"
        self.pro = self._init_tushare()
    
    def _init_tushare(self):
        """初始化 Tushare Pro"""
        try:
            import tushare as ts
            token = os.getenv("TUSHARE_TOKEN")
            if not token:
                raise ValueError("未设置 TUSHARE_TOKEN 环境变量")
            ts.set_token(token)
            return ts.pro_api()
        except ImportError:
            raise ImportError("请安装 tushare: pip install tushare")
    
    @cached(ttl=300, key_prefix="tushare_index")
    def _fetch_data(self, ts_code: str) -> Dict[str, Any]:
        """从 Tushare 获取数据（带缓存）"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        df = self.pro.index_daily(
            ts_code=ts_code,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        if df is None or df.empty:
            raise Exception(f"未找到指数 {ts_code} 数据")
        
        # 获取最新数据
        latest = df.iloc[0]  # Tushare 返回的是倒序，最新在前面
        
        return {
            "code": ts_code.split('.')[0],
            "name": self.INDICES[ts_code]["name"],
            "date": latest['trade_date'],
            "close": float(latest['close']),
            "open": float(latest['open']),
            "high": float(latest['high']),
            "low": float(latest['low']),
            "change": float(latest['pct_chg']) if 'pct_chg' in latest else 0,
            "change_amount": float(latest['change']) if 'change' in latest else 0,
            "volume": float(latest['vol']) if 'vol' in latest else 0,
            "amount": float(latest['amount']) if 'amount' in latest else 0,
        }
    
    def fetch(self, index_code: Optional[str] = None) -> Dict[str, Any]:
        """获取大盘指数数据"""
        try:
            indices_data = []
            
            for ts_code in self.INDICES.keys():
                try:
                    data = self._fetch_data(ts_code)
                    indices_data.append(data)
                except Exception as e:
                    print(f"  [WARN] 获取指数 {ts_code} 失败: {e}")
                    continue
            
            if not indices_data:
                raise Exception("无法获取任何指数数据")
            
            evaluation = self._generate_evaluation(indices_data)
            
            return {
                "status": "success",
                "indices": indices_data,
                "evaluation": evaluation
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "indices": [],
                "evaluation": f"❌ 获取指数数据失败: {e}"
            }
    
    def _generate_evaluation(self, indices_data: List[Dict]) -> str:
        """生成大盘指数评价文本"""
        lines = ["## 📈 大盘指数行情\n"]
        
        # 按重要性排序
        priority = {"上证指数": 1, "深证成指": 2, "创业板指": 3, "沪深300": 4, "科创50": 5}
        indices_data.sort(key=lambda x: priority.get(x["name"], 99))
        
        for idx in indices_data:
            change_emoji = "📈" if idx["change"] > 0 else "📉" if idx["change"] < 0 else "➡️"
            lines.append(f"### {idx['name']} ({idx['code']})")
            lines.append(f"- **收盘**: {idx['close']:.2f} {change_emoji}")
            lines.append(f"- **涨跌**: {idx['change']:+.2f}% ({idx['change_amount']:+.2f})")
            lines.append(f"- **最高**: {idx['high']:.2f} / **最低**: {idx['low']:.2f}")
            lines.append(f"- **成交量**: {idx['volume']/10000:.2f}万手")
            lines.append("")
        
        # 整体市场判断
        sh_index = next((x for x in indices_data if x["code"] == "000001"), None)
        cy_index = next((x for x in indices_data if x["code"] == "399006"), None)
        
        if sh_index and cy_index:
            lines.append("### 市场综述")
            sh_change = sh_index["change"]
            cy_change = cy_index["change"]
            
            if sh_change > 1 and cy_change > 1:
                trend = "强势上涨"
                desc = "大盘整体表现强势，主板与创业板同步上涨，市场情绪积极"
            elif sh_change > 0 and cy_change > 0:
                trend = "震荡上涨"
                desc = "大盘整体上涨，但涨幅温和，市场谨慎乐观"
            elif sh_change < -1 and cy_change < -1:
                trend = "大幅回调"
                desc = "大盘整体下跌，主板与创业板同步走弱，需注意风险"
            elif sh_change < 0 and cy_change < 0:
                trend = "震荡调整"
                desc = "大盘小幅回调，市场处于调整阶段，观望为主"
            elif abs(sh_change - cy_change) > 1:
                trend = "分化走势"
                if sh_change > cy_change:
                    desc = "主板强于创业板，大盘股表现较好，中小创相对弱势"
                else:
                    desc = "创业板强于主板，成长风格占优，题材股活跃"
            else:
                trend = "窄幅震荡"
                desc = "大盘波动较小，市场等待方向选择"
            
            lines.append(f"- **整体趋势**: {trend}")
            lines.append(f"- **市场判断**: {desc}")
        
        return "\n".join(lines)
    
    @classmethod
    def as_node(cls):
        """创建 LangGraph 节点函数"""
        agent = cls()
        
        def index_node(state: AgentState) -> Dict[str, Any]:
            print("    [IndexAgent] 正在获取大盘指数数据（Tushare）...", flush=True)
            result = agent.fetch()
            
            if result["status"] == "success":
                print(f"    [IndexAgent] 获取成功，指数数量: {len(result.get('indices', []))}", flush=True)
            else:
                print(f"    [IndexAgent] 获取失败: {result.get('error', '未知错误')}", flush=True)
            
            if result["status"] != "success":
                return {
                    "index_result": result["evaluation"],
                    "index_sources": []
                }
            
            return {
                "index_result": result["evaluation"],
                "index_sources": [{
                    "source": "tushare",
                    "index_count": len(result['indices']),
                    "data_date": result['indices'][0]['date'] if result['indices'] else None
                }]
            }
        
        return index_node
