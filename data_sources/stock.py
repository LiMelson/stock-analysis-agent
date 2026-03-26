"""
个股数据源模块 - 基于 Tushare Pro

提供个股相关数据
注册地址：https://tushare.pro/register
"""

import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.state import AgentState
from configs.cache import cached


class StockDataSource:
    """
    个股数据源 - 基于 Tushare Pro
    """
    
    def __init__(self):
        self.name = "stock"
        self.description = "个股数据（Tushare Pro）"
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
    
    def _extract_stock_code(self, query: str) -> Optional[str]:
        """从查询文本中提取股票代码"""
        # 匹配6位数字股票代码
        patterns = [
            r'(\d{6})',
            r'(sh|sz|bj)(\d{6})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                if len(match.groups()) == 2:
                    return match.group(2)
                return match.group(1)
        
        # 常见股票名称映射
        name_map = {
            "茅台": "600519", "贵州茅台": "600519",
            "平安": "000001", "中国平安": "601318",
            "宁德": "300750", "宁德时代": "300750",
            "比亚迪": "002594",
            "招行": "600036", "招商银行": "600036",
            "五粮液": "000858",
            "中免": "601888", "中国中免": "601888",
            "美的": "000333", "美的集团": "000333",
            "格力": "000651", "格力电器": "000651",
            "中兴": "000063", "中兴通讯": "000063",
            "海康": "002415", "海康威视": "002415",
            "立讯": "002475", "立讯精密": "002475",
            "隆基": "601012", "隆基绿能": "601012",
            "迈瑞": "300760", "迈瑞医疗": "300760",
            "药明": "603259", "药明康德": "603259",
            "恒瑞": "600276", "恒瑞医药": "600276",
            "伊利": "600887", "伊利股份": "600887",
            "中信证券": "600030",
            "东方财富": "300059",
            "工业富联": "601138",
            "中芯国际": "688981",
        }
        
        for name, code in name_map.items():
            if name in query:
                return code
        
        return None
    
    def _get_ts_code(self, stock_code: str) -> str:
        """转换为 Tushare 格式（添加.SZ/.SH后缀）"""
        if stock_code.startswith('6'):
            return f"{stock_code}.SH"
        else:
            return f"{stock_code}.SZ"
    
    @cached(ttl=300, key_prefix="tushare_stock_daily")
    def _fetch_daily_data(self, ts_code: str) -> Dict[str, Any]:
        """获取日线数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = self.pro.daily(
            ts_code=ts_code,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        if df is None or df.empty:
            raise Exception(f"未找到股票 {ts_code} 数据")
        
        latest = df.iloc[0]
        
        return {
            "code": ts_code.split('.')[0],
            "name": self._get_stock_name(ts_code.split('.')[0]),
            "price": float(latest['close']),
            "change": float(latest['pct_chg']),
            "change_amount": float(latest['change']),
            "open": float(latest['open']),
            "high": float(latest['high']),
            "low": float(latest['low']),
            "pre_close": float(latest['pre_close']),
            "volume": float(latest['vol']),
            "amount": float(latest['amount']),
        }
    
    def _get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        try:
            df = self.pro.stock_basic(ts_code=self._get_ts_code(code), fields='name')
            if df is not None and not df.empty:
                return df.iloc[0]['name']
        except:
            pass
        return "未知"
    
    @cached(ttl=600, key_prefix="tushare_stock_hist")
    def _fetch_history_data(self, ts_code: str, days: int = 20) -> Dict[str, Any]:
        """获取历史K线数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)
        
        df = self.pro.daily(
            ts_code=ts_code,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        if df is None or df.empty:
            raise Exception(f"未找到股票 {ts_code} 的历史数据")
        
        df = df.head(days)
        
        klines = []
        for _, row in df.iterrows():
            klines.append({
                "date": row['trade_date'],
                "open": float(row['open']),
                "close": float(row['close']),
                "high": float(row['high']),
                "low": float(row['low']),
                "volume": float(row['vol']),
                "amount": float(row['amount']),
                "change": float(row['pct_chg']),
            })
        
        closes = [k['close'] for k in klines]
        avg_5 = sum(closes[-5:]) / len(closes[-5:]) if len(closes) >= 5 else None
        avg_10 = sum(closes[-10:]) / len(closes[-10:]) if len(closes) >= 10 else None
        avg_20 = sum(closes[-20:]) / len(closes[-20:]) if len(closes) >= 20 else None
        
        return {
            "code": ts_code.split('.')[0],
            "klines": klines,
            "tech": {
                "ma5": avg_5,
                "ma10": avg_10,
                "ma20": avg_20,
            }
        }
    
    @cached(ttl=3600, key_prefix="tushare_stock_fin")
    def _fetch_financial_data(self, ts_code: str) -> Dict[str, Any]:
        """获取财务数据"""
        # 获取最新财务指标
        df = self.pro.fina_indicator(ts_code=ts_code, limit=1)
        
        if df is None or df.empty:
            raise Exception(f"未找到股票 {ts_code} 的财务数据")
        
        latest = df.iloc[0]
        
        return {
            "code": ts_code.split('.')[0],
            "report_date": latest.get('end_date', '未知'),
            "eps": float(latest.get('eps', 0)) if latest.get('eps') else None,
            "bvps": float(latest.get('bps', 0)) if latest.get('bps') else None,
            "roe": float(latest.get('roe', 0)) if latest.get('roe') else None,
            "revenue_growth": float(latest.get('q_sales_yoy', 0)) if latest.get('q_sales_yoy') else None,
            "profit_growth": float(latest.get('q_profit_yoy', 0)) if latest.get('q_profit_yoy') else None,
            "debt_ratio": float(latest.get('debt_to_assets', 0)) if latest.get('debt_to_assets') else None,
        }
    
    def fetch(self, query: str) -> Dict[str, Any]:
        """获取个股综合数据"""
        stock_code = self._extract_stock_code(query)
        
        if not stock_code:
            return {
                "status": "error",
                "error": "无法从查询中提取股票代码",
                "evaluation": "请在问题中提供股票代码（如：600519）或股票名称（如：贵州茅台）"
            }
        
        ts_code = self._get_ts_code(stock_code)
        
        try:
            # 获取日线数据
            daily = self._fetch_daily_data(ts_code)
            
            # 获取历史数据
            try:
                history = self._fetch_history_data(ts_code, days=20)
            except Exception as e:
                print(f"  [WARN] 获取历史数据失败: {e}")
                history = None
            
            # 获取财务数据（可选）
            try:
                financial = self._fetch_financial_data(ts_code)
            except Exception as e:
                print(f"  [WARN] 获取财务数据失败: {e}")
                financial = None
            
            evaluation = self._generate_evaluation(daily, history, financial)
            
            return {
                "status": "success",
                "code": stock_code,
                "name": daily.get("name", "未知"),
                "realtime": daily,
                "history": history,
                "financial": financial,
                "evaluation": evaluation
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "code": stock_code,
                "evaluation": f"获取股票 {stock_code} 数据失败: {e}"
            }
    
    def _generate_evaluation(self, daily: Dict, history: Optional[Dict], financial: Optional[Dict]) -> str:
        """生成个股评价文本"""
        name = daily.get("name", "未知")
        code = daily.get("code", "")
        price = daily.get("price", 0)
        change = daily.get("change", 0)
        change_amount = daily.get("change_amount", 0)
        
        change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        lines = [f"## 个股行情 - {name}({code})\n"]
        lines.append(f"### 实时行情 {change_emoji}")
        lines.append(f"- **最新价**: ¥{price:.2f}")
        lines.append(f"- **涨跌**: {change:+.2f}% ({change_amount:+.2f})")
        lines.append(f"- **今开**: ¥{daily.get('open', 0):.2f} / **昨收**: ¥{daily.get('pre_close', 0):.2f}")
        lines.append(f"- **最高**: ¥{daily.get('high', 0):.2f} / **最低**: ¥{daily.get('low', 0):.2f}")
        lines.append(f"- **成交量**: {daily.get('volume', 0)/10000:.2f}万手")
        lines.append("")
        
        # 技术分析
        if history and history.get("tech"):
            tech = history["tech"]
            ma5 = tech.get("ma5")
            ma10 = tech.get("ma10")
            ma20 = tech.get("ma20")
            
            lines.append("### 技术指标")
            if ma5:
                lines.append(f"- **MA5**: ¥{ma5:.2f}")
            if ma10:
                lines.append(f"- **MA10**: ¥{ma10:.2f}")
            if ma20:
                lines.append(f"- **MA20**: ¥{ma20:.2f}")
            
            if ma5 and ma10 and ma20:
                if price > ma5 > ma10 > ma20:
                    trend = "多头排列"
                elif price < ma5 < ma10 < ma20:
                    trend = "空头排列"
                else:
                    trend = "震荡整理"
                lines.append(f"- **趋势判断**: {trend}")
            
            lines.append("")
        
        # 财务简评
        if financial:
            lines.append("### 财务简览")
            roe = financial.get("roe")
            if roe:
                lines.append(f"- **ROE**: {roe:.2f}%")
            lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    def as_node(cls):
        """创建 LangGraph 节点函数"""
        agent = cls()
        
        def stock_node(state: AgentState) -> Dict[str, Any]:
            print("    [StockAgent] 正在获取个股数据（Tushare）...", flush=True)
            
            query = state.get("stock_query", "")
            if not query:
                query = state.get("question", "")
            
            if not query:
                return {
                    "stock_result": "未提供股票查询",
                    "stock_sources": []
                }
            
            result = agent.fetch(query)
            
            if result["status"] != "success":
                error_msg = result.get("error", "获取个股数据失败")
                print(f"    [StockAgent] 获取失败: {error_msg}", flush=True)
                return {
                    "stock_result": f"获取个股数据失败: {error_msg}",
                    "stock_sources": []
                }
            
            print(f"    [StockAgent] 获取成功: {result.get('name', 'Unknown')}", flush=True)
            
            sources = [{
                "source": "tushare",
                "stock_code": result["code"],
                "stock_name": result["name"],
                "data_type": "daily"
            }]
            
            if result.get("history"):
                sources.append({"source": "tushare", "stock_code": result["code"], "data_type": "history"})
            
            if result.get("financial"):
                sources.append({"source": "tushare", "stock_code": result["code"], "data_type": "financial"})
            
            return {
                "stock_result": result["evaluation"],
                "stock_sources": sources
            }
        
        return stock_node
