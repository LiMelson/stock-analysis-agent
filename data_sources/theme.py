"""
题材/板块数据源模块 - 基于 Tushare Pro

使用 Tushare Pro 的板块数据
注册地址：https://tushare.pro/register
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from core.state import AgentState
from configs.cache import cached


class ThemeDataSource:
    """
    题材/板块数据源 - 基于 Tushare Pro
    """
    
    def __init__(self):
        self.name = "theme"
        self.description = "题材板块数据（Tushare Pro 同花顺板块）"
        self.pro = self._init_tushare()
    
    def _init_tushare(self):
        """初始化 Tushare Pro"""
        try:
            import tushare as ts
            token = os.getenv("TUSHARE_TOKEN")
            if not token or token == "你的token_here":
                raise ValueError("请在 .env 文件中设置正确的 TUSHARE_TOKEN")
            ts.set_token(token)
            return ts.pro_api()
        except ImportError:
            raise ImportError("请安装 tushare: pip install tushare")
    
    @cached(ttl=300, key_prefix="tushare_theme_hot")
    def _fetch_hot_sectors(self, top_n: int = 10) -> Dict[str, Any]:
        """获取热点板块"""
        trade_date = datetime.now().strftime('%Y%m%d')
        
        # 获取同花顺板块指数日线（按涨跌幅排序）
        df = self.pro.ths_daily(trade_date=trade_date, fields='ts_code,name,close,open,high,low,pct_change')
        
        if df is None or df.empty:
            raise Exception("无法获取板块数据")
        
        # 按涨跌幅排序
        df = df.sort_values('pct_change', ascending=False)
        
        sectors = []
        for _, row in df.head(top_n).iterrows():
            sectors.append({
                "name": row['name'],
                "change": float(row['pct_change']) if 'pct_change' in row else 0,
                "close": float(row['close']),
            })
        
        return {"status": "success", "sectors": sectors}
    
    @cached(ttl=300, key_prefix="tushare_theme_concept")
    def _fetch_concept_boards(self, top_n: int = 10) -> Dict[str, Any]:
        """获取概念板块"""
        # 使用概念指数数据
        trade_date = datetime.now().strftime('%Y%m%d')
        
        # 获取概念指数列表
        df_list = self.pro.ths_index()
        if df_list is None or df_list.empty:
            raise Exception("无法获取概念板块列表")
        
        # 获取前N个概念的日线数据
        concepts = []
        count = 0
        for _, row in df_list.iterrows():
            if count >= top_n:
                break
            try:
                ts_code = row['ts_code']
                name = row['name']
                df = self.pro.ths_daily(ts_code=ts_code, trade_date=trade_date)
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    concepts.append({
                        "name": name,
                        "change": float(latest.get('pct_change', 0)),
                    })
                    count += 1
            except:
                continue
        
        return {"status": "success", "concepts": concepts}
    
    def fetch(self, query: str = "") -> Dict[str, Any]:
        """获取题材板块综合数据"""
        try:
            # 获取热点板块
            hot_sectors = self._fetch_hot_sectors(top_n=10)
            
            # 获取概念板块（简化，避免过多请求）
            try:
                hot_concepts = self._fetch_concept_boards(top_n=5)
            except Exception as e:
                print(f"  [WARN] 获取概念板块失败: {e}")
                hot_concepts = {"status": "error", "concepts": []}
            
            # 生成评价文本
            evaluation = self._generate_evaluation(hot_sectors, hot_concepts)
            
            return {
                "status": "success",
                "hot_sectors": hot_sectors.get("sectors", []),
                "hot_concepts": hot_concepts.get("concepts", []),
                "evaluation": evaluation
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "evaluation": f"获取板块数据失败: {e}"
            }
    
    def _generate_evaluation(self, hot_sectors: Dict, hot_concepts: Dict) -> str:
        """生成板块评价文本"""
        lines = ["## 题材板块热点\n"]
        
        # 热点行业板块
        if hot_sectors.get("sectors"):
            lines.append("### 热门板块（涨幅前5）")
            for i, sector in enumerate(hot_sectors["sectors"][:5], 1):
                change_emoji = "📈" if sector['change'] > 0 else "📉"
                lines.append(f"{i}. **{sector['name']}** {change_emoji} {sector['change']:+.2f}%")
            lines.append("")
        
        # 热点概念
        if hot_concepts.get("concepts"):
            lines.append("### 热门概念")
            for i, concept in enumerate(hot_concepts["concepts"][:5], 1):
                change_emoji = "📈" if concept['change'] > 0 else "📉"
                lines.append(f"{i}. **{concept['name']}** {change_emoji} {concept['change']:+.2f}%")
            lines.append("")
        
        # 热点总结
        lines.append("### 热点总结")
        top_sector = hot_sectors["sectors"][0] if hot_sectors.get("sectors") else None
        
        if top_sector:
            if top_sector["change"] > 5:
                lines.append("- 今日市场热点明确，部分板块出现涨停潮")
            elif top_sector["change"] > 2:
                lines.append("- 市场热点活跃，板块轮动明显")
            else:
                lines.append("- 市场热点分散，赚钱效应一般")
            
            lines.append(f"- 最强板块: {top_sector['name']} (+{top_sector['change']:.2f}%)")
        
        return "\n".join(lines)
    
    @classmethod
    def as_node(cls):
        """创建 LangGraph 节点函数"""
        agent = cls()
        
        def theme_node(state: AgentState) -> Dict[str, Any]:
            print("    [ThemeAgent] 正在获取题材板块数据（Tushare）...", flush=True)
            
            query = state.get("theme_query", "")
            if not query:
                query = state.get("question", "")
            
            result = agent.fetch(query)
            
            if result["status"] == "success":
                print(f"    [ThemeAgent] 获取成功，板块数: {len(result.get('hot_sectors', []))}", flush=True)
            else:
                print(f"    [ThemeAgent] 获取失败: {result.get('error', '未知错误')}", flush=True)
            
            if result["status"] != "success":
                return {
                    "theme_result": result["evaluation"],
                    "theme_sources": []
                }
            
            return {
                "theme_result": result["evaluation"],
                "theme_sources": [{
                    "source": "tushare",
                    "data_type": "ths_sector",
                    "sector_count": len(result.get('hot_sectors', []))
                }]
            }
        
        return theme_node
