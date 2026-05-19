# -*- coding: utf-8 -*-

import os
import re
import gc
import sys
import time
import math
import queue
import pickle
import random
import logging
import datetime
from collections import deque
from dataclasses import dataclass
import sqlite3
import datetime
import pandas as pd
import numpy as np
import requests
from fake_useragent import UserAgent

# 可选：机器学习
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump, load
import os
print(f"当前工作目录 (Current Working Directory): {os.getcwd()}")
# ... 您的其他代码
# ==============================================================================
# 0. 日志 logging
# ==============================================================================
LOG_DIR = os.path.abspath("./logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("etf_bot")
logger.setLevel(logging.DEBUG)

# 控制台
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))

# 文件
fh = logging.FileHandler(os.path.join(LOG_DIR, "trade.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S"
))

# 避免重复添加
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

# ==============================================================================
# 1. 反爬 & 会话
# ==============================================================================
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['no_proxy'] = '127.0.0.1,localhost,finance.eastmoney.com,gtimg.cn,sina.com.cn'

ua = UserAgent()


def get_super_headers():
    return {
        "User-Agent": ua.random,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": f"https://finance.eastmoney.com/etf/{random.randint(150000, 160000)}.html",
        "Cookie": f"EMFUND1={int(time.time())}; EMFUND2=null; qgqp_b_id={random.getrandbits(64)}; _adsame_fullscreen_18900={random.randint(0, 1)}",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": f"max-age={random.randint(0, 30)}",
        "TE": "trailers"
    }


original_get = requests.get


def ultra_safe_request(url, params=None, **kwargs):
    if any(domain in url for domain in ["eastmoney.com", "emdata.cn", "gtimg.cn", "sina.com.cn"]):
        kwargs['headers'] = get_super_headers()
        kwargs['timeout'] = (10, 20)
        kwargs['verify'] = random.choice([True, False])
        delay = random.uniform(1.0, 3.0) if not kwargs.get('is_retry', False) else random.uniform(5.0, 10.0)
        time.sleep(delay)
    return original_get(url, params=params, **kwargs)


requests.get = ultra_safe_request


# ==============================================================================
# 2. 状态持久化
# ==============================================================================
class StateManager:
    def __init__(self, symbol, state_file="trading_state.pkl"):
        self.state_file = state_file
        self.symbol = symbol
        self.default_state = {
            "account": {
                "cash": 100000.0,
                "total_asset": 100000.0,
                "position": 0,  # 持仓股数
                "latest_market_value": 0.0,
                "avg_cost": 0.0  # 新增：平均成本
            },
            "strategy": {
                "price_history": [],
                "atr_history": [],
                "position": 0,  # 1=多仓, -1=空仓, 0=空仓
                "entry_price": 0.0,
                "last_trade_time": None,
                "consecutive_losses": 0,  # 新增：连续亏损次数
                "total_trades": 0,  # 新增：总交易次数
                "winning_trades": 0  # 新增：盈利交易次数
            },
            "system": {
                "last_update_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_count": 0,
                "transaction_history": []
            }
        }

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                if "symbol" in state and state["symbol"] == self.symbol:
                    # 数据完整性验证和修复
                    if state["account"]["cash"] < 0:
                        logger.warning(f"检测到负现金余额 {state['account']['cash']:.2f}，重置为初始资金")
                        state["account"]["cash"] = 100000.0
                        state["account"]["position"] = 0
                        state["strategy"]["position"] = 0

                    logger.info(
                        f"成功加载历史状态 | 上次更新时间: {state['system']['last_update_time']} | 运行次数: {state['system']['run_count'] + 1}"
                    )
                    logger.info(
                        f"持仓股数: {state['account']['position']} | 总资产: {state['account']['total_asset']:.2f} | 现金: {state['account']['cash']:.2f}"
                    )
                    return state
                else:
                    logger.warning("状态文件标的不匹配，初始化新状态")
                    return self._init_new_state()
            except Exception as e:
                logger.exception(f"加载状态失败，初始化新状态 | {e}")
                return self._init_new_state()
        else:
            logger.info("未找到历史状态，初始化新状态")
            return self._init_new_state()

    def _init_new_state(self):
        new_state = self.default_state.copy()
        new_state["symbol"] = self.symbol
        return new_state

    def save_state(self, account, strategy, transaction=None):
        try:
            prev_state = self.load_state() if os.path.exists(self.state_file) else self.default_state
            state = {
                "symbol": self.symbol,
                "account": account,
                "strategy": strategy,
                "system": {
                    "last_update_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run_count": prev_state["system"]["run_count"] + 1,
                    "transaction_history": prev_state["system"]["transaction_history"] + (
                        [transaction] if transaction else []
                    )
                }
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
            return True
        except Exception as e:
            logger.exception(f"保存状态失败: {e}")
            return False


# ==============================================================================
# 2.5. 数据库管理 (新增)
# ==============================================================================
class DBManager:
    def __init__(self, db_path="etf_data.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        """建立数据库连接"""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化数据库和表结构"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.commit()

    def read_data(self, symbol: str) -> pd.DataFrame:
        """从数据库读取指定标的的历史数据"""
        with self._get_conn() as conn:
            # 读取时按日期排序
            df = pd.read_sql_query(f"SELECT * FROM klines WHERE symbol = '{symbol}' ORDER BY date ASC", conn)
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
        logger.info(f"从数据库加载了 {len(df)} 条 {symbol} 的历史数据")
        return df

    def write_data(self, df: pd.DataFrame, symbol: str):
        """将新的数据写入数据库，如果已存在则替换"""
        if df.empty:
            return

        with self._get_conn() as conn:
            # 确保 DataFrame 包含 symbol 列
            df_to_write = df.copy()
            df_to_write['symbol'] = symbol

            # 将 date 转为字符串以匹配数据库 TEXT 类型
            df_to_write['date'] = df_to_write['date'].dt.strftime('%Y-%m-%d')

            # 使用 to_sql 的 "replace" 模式来插入或更新数据
            # if_exists='append' 配合主键可以实现 INSERT OR REPLACE 的效果
            # 但 pandas 的 to_sql 在处理复合主键时行为不一，我们手动实现
            cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            df_to_write = df_to_write[cols]

            cursor = conn.cursor()
            for row in df_to_write.itertuples(index=False):
                cursor.execute(f"""
                    INSERT OR REPLACE INTO klines ({', '.join(cols)})
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, row)

            conn.commit()
            logger.info(f"成功向数据库写入/更新了 {len(df_to_write)} 条 {symbol} 的数据")

# ==============================================================================
# 3. 指标与工具
# ==============================================================================
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=0)
    z = (series - m) / s.replace(0, np.nan)
    return z.fillna(0)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def true_range(high, low, prev_close):
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # 兼容只有 close 的情况
    if {"high", "low", "close"}.issubset(set(map(str.lower, df.columns))):
        # 统一列名
        cols = {c.lower(): c for c in df.columns}
        H, L, C = df[cols["high"]].astype(float), df[cols["low"]].astype(float), df[cols["close"]].astype(float)
        prev_close = C.shift(1)
        tr = true_range(H, L, prev_close)
        atr = tr.rolling(window).mean()
        return atr.fillna(method="bfill").fillna(0.0)
    else:
        # 退化到 |close - prev_close|
        close_col = None
        for cand in ["close", "收盘"]:
            if cand in df.columns: close_col = cand
        if close_col is None:
            close_col = df.columns[-1]
        close = pd.to_numeric(df[close_col], errors="coerce")
        tr = (close - close.shift(1)).abs()
        return tr.rolling(window).mean().fillna(method="bfill").fillna(0.0)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0):
    if returns.std(ddof=0) == 0: return 0.0
    return (returns.mean() - risk_free) / returns.std(ddof=0) * math.sqrt(252)


def max_drawdown(equity_curve: pd.Series):
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return drawdown.min()


def is_market_open_cn():
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    morning_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    morning_close = now.replace(hour=11, minute=30, second=0, microsecond=0)
    afternoon_open = now.replace(hour=13, minute=0, second=0, microsecond=0)
    afternoon_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return (morning_open <= now <= morning_close) or (afternoon_open <= now <= afternoon_close)


# ==============================================================================
# 4. 事件
# ==============================================================================
class Event: ...


class MarketEvent(Event):
    def __init__(self, timestamp, symbol, latest_price):
        self.type, self.timestamp, self.symbol, self.latest_price = 'MARKET', timestamp, symbol, latest_price


class SignalEvent(Event):
    def __init__(self, timestamp, symbol, direction, strength=1.0):
        self.type, self.timestamp, self.symbol, self.direction, self.strength = 'SIGNAL', timestamp, symbol, direction, strength


class OrderEvent(Event):
    def __init__(self, timestamp, symbol, order_type, quantity, direction):
        self.type, self.timestamp, self.symbol, self.order_type, self.quantity, self.direction = 'ORDER', timestamp, symbol, order_type, quantity, direction
        self.latest_price = None


class FillEvent(Event):
    def __init__(self, timestamp, symbol, quantity, direction, fill_price, commission):
        self.type, self.timestamp, self.symbol, self.quantity, self.direction, self.fill_price, self.commission = 'FILL', timestamp, symbol, quantity, direction, fill_price, commission
        logger.info(
            f"成交 | {self.direction} {self.quantity} 股 {self.symbol} @ {self.fill_price:.3f} | 手续费: {self.commission:.2f} RMB")
        self.transaction_record = {
            "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price": fill_price,
            "commission": commission,
            "total": fill_price * quantity + (commission if direction == 'BUY' else -commission)
        }


# ==============================================================================
# 5. 数据源
# ==============================================================================
class MultiSourceDataHandler:
    def __init__(self, events, symbol):
        import akshare as ak
        self.ak = ak
        self.events = events
        self.symbol = symbol
        self.symbol_no_suffix = symbol.replace(".SZ", "").replace(".SH", "")
        self.failure_counts = {"akshare_em": 0, "akshare_sina": 0, "tencent": 0}
        self.max_fail = 3
        self.cooldowns = {}
        self.base_interval = 30

    def _is_source_available(self, source):
        return time.time() > self.cooldowns.get(source, 0)

    def _fetch_akshare_em(self):
        if not self._is_source_available("akshare_em"): return None
        try:
            df = self.ak.fund_etf_spot_em()
            row = df[df["代码"] == self.symbol_no_suffix]
            if not row.empty and pd.notna(row["最新价"].iloc[0]):
                self.failure_counts["akshare_em"] = 0
                return {"price": float(row["最新价"].iloc[0]), "source": "东方财富(em)"}
        except Exception as e:
            self.failure_counts["akshare_em"] += 1
            if self.failure_counts["akshare_em"] >= self.max_fail:
                self.cooldowns["akshare_em"] = time.time() + 300
                logger.warning(f"东方财富(em)连续失败，冷却5分钟 | {str(e)[:120]}")
        return None

    def _fetch_akshare_sina(self):
        if not self._is_source_available("akshare_sina"): return None
        try:
            df = self.ak.fund_etf_spot_sina(symbol=self.symbol)
            if not df.empty and "现价" in df.columns and pd.notna(df["现价"].iloc[0]):
                self.failure_counts["akshare_sina"] = 0
                return {"price": float(df["现价"].iloc[0]), "source": "新浪财经(sina)"}
        except Exception as e:
            self.failure_counts["akshare_sina"] += 1
            if self.failure_counts["akshare_sina"] >= self.max_fail:
                self.cooldowns["akshare_sina"] = time.time() + 300
                logger.warning(f"新浪财经连续失败，冷却5分钟 | {str(e)[:120]}")
        return None

    def _fetch_tencent(self):
        if not self._is_source_available("tencent"): return None
        try:
            market = "sz" if ".SZ" in self.symbol else "sh"
            url = f"https://qt.gtimg.cn/q={market}{self.symbol_no_suffix}"
            response = requests.get(url, headers=get_super_headers(), timeout=10)
            data = response.text.split("~")
            if len(data) > 4 and data[3] and re.match(r"^\d+(\.\d+)?$", data[3]):
                price = float(data[3])
                if price > 0:
                    self.failure_counts["tencent"] = 0
                    return {"price": price, "source": "腾讯财经(gtimg)"}
        except Exception as e:
            self.failure_counts["tencent"] += 1
            if self.failure_counts["tencent"] >= self.max_fail:
                self.cooldowns["tencent"] = time.time() + 300
                logger.warning(f"腾讯财经连续失败，冷却5分钟 | {str(e)[:120]}")
        return None

    def update_bars(self):
        sources = sorted(self.failure_counts.keys(), key=lambda k: self.failure_counts[k])
        fetch_methods = {
            "akshare_em": self._fetch_akshare_em,
            "akshare_sina": self._fetch_akshare_sina,
            "tencent": self._fetch_tencent
        }
        for source in sources:
            data = fetch_methods[source]()
            if data:
                logger.info(f"最新价: {data['price']:.3f} (来源: {data['source']})")
                self.events.put(MarketEvent(datetime.datetime.now(), self.symbol, data['price']))
                return True

        sleep_time = random.randint(20, 60)
        logger.warning(f"所有数据源失败，休眠 {sleep_time} 秒...")
        time.sleep(sleep_time)
        return False

    def get_random_interval(self):
        return self.base_interval + random.uniform(-10, 10)


# ==============================================================================
# 6. ML 特征工程 & 训练
# ==============================================================================
class MLSignalFilter:
    """
    用历史数据训练分类器，预测「开仓后 N 天收益是否达标/回撤是否可控」。
    用作信号过滤器：只有在 ML 置信度 >= 阈值时才允许下单。
    """

    def __init__(self, model_path="ml_filter.joblib", n_days_take=5, ret_target=0.006, risk_cap=0.04):
        self.model_path = model_path
        self.n_days_take = n_days_take
        self.ret_target = ret_target  # 未来 n 天最大高点超出开仓价的涨幅阈值
        self.risk_cap = risk_cap  # 未来 n 天最大回撤不超过该阈值（相对于开仓）
        self.clf = None
        self.feature_cols = None
        self.threshold = 0.65  # 提高预测概率阈值，更保守

    def build_dataset(self, df: pd.DataFrame):
        """
        df: 必须至少包含列 ['date','close']，最好包含 high/low。
        """
        X = pd.DataFrame(index=df.index)
        close = df['close'].astype(float)
        X["z30"] = rolling_zscore(close, 30)
        X["z60"] = rolling_zscore(close, 60)
        X["ret1"] = close.pct_change(1).fillna(0)
        X["ret5"] = close.pct_change(5).fillna(0)
        X["ret10"] = close.pct_change(10).fillna(0)  # 新增：更长期动量
        X["rsi14"] = compute_rsi(close, 14)
        X["vol20"] = close.pct_change().rolling(20).std().fillna(0)
        X["vol_ratio"] = X["vol20"] / X["vol20"].rolling(60).mean().fillna(1)  # 新增：波动率比率
        X["atr14"] = compute_atr(df, 14)
        X["atr_pct"] = (X["atr14"] / close.replace(0, np.nan)).fillna(0)

        # 趋势指标
        X["sma5"] = close.rolling(5).mean()
        X["sma20"] = close.rolling(20).mean()
        X["trend"] = (X["sma5"] / X["sma20"] - 1).fillna(0)  # 新增：短期vs长期趋势

        # 目标：未来 n 天最高价相对开仓（t+1）涨幅是否达到 ret_target 且不触发 risk_cap 回撤
        if "high" in df.columns and "low" in df.columns:
            future_max = df["high"].shift(-1).rolling(self.n_days_take).max()
            future_min = df["low"].shift(-1).rolling(self.n_days_take).min()
        else:
            future_max = close.shift(-1).rolling(self.n_days_take).max()
            future_min = close.shift(-1).rolling(self.n_days_take).min()

        entry = close.shift(1)
        up_ok = (future_max / entry - 1.0) >= self.ret_target
        dd_ok = (future_min / entry - 1.0) >= -self.risk_cap
        y = (up_ok & dd_ok).astype(int)

        # 对齐
        data = X.copy()
        data["y"] = y
        data = data.dropna().copy()
        self.feature_cols = [c for c in data.columns if c != "y"]
        return data[self.feature_cols], data["y"]

    def train(self, df_hist: pd.DataFrame, threshold=0.65):
        self.threshold = threshold
        X, y = self.build_dataset(df_hist)
        if len(X) < 200:
            logger.warning("历史样本太少，跳过 ML 训练。")
            return None

        tscv = TimeSeriesSplit(n_splits=5)
        best_auc, best_model = -1, None

        # 轻量级随机搜索
        for i in range(20):
            params = {
                "n_estimators": random.choice([200, 300, 400]),
                "max_depth": random.choice([3, 4, 5, 6, None]),
                "min_samples_split": random.choice([2, 5, 10]),
                "min_samples_leaf": random.choice([1, 2, 4]),
                "max_features": random.choice(["sqrt", "log2", None]),
                "bootstrap": random.choice([True, False]),
                "random_state": 42 + i
            }
            clf = RandomForestClassifier(**params)
            aucs = []
            for tr_idx, te_idx in tscv.split(X):
                Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
                ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
                clf.fit(Xtr, ytr)
                proba = clf.predict_proba(Xte)[:, 1]
                aucs.append(roc_auc_score(yte, proba))
            avg_auc = float(np.mean(aucs))
            if avg_auc > best_auc:
                best_auc, best_model = avg_auc, clf

        if best_model is not None:
            best_model.fit(X, y)
            self.clf = best_model
            dump({"model": self.clf, "features": self.feature_cols, "threshold": self.threshold}, self.model_path)
            logger.info(f"ML 训练完成 | AUC={best_auc:.4f} | 模型已保存: {self.model_path}")
        else:
            logger.warning("ML 随机搜索未找到更优模型。")

    def load(self):
        if os.path.exists(self.model_path):
            pack = load(self.model_path)
            self.clf = pack["model"]
            self.feature_cols = pack["features"]
            self.threshold = pack.get("threshold", 0.65)
            logger.info(f"已加载 ML 模型: {self.model_path} | 阈值={self.threshold}")
            return True
        return False

    def allow_trade(self, df_recent: pd.DataFrame) -> bool:
        """
        用最近一条样本做推断，若 proba >= threshold 返回 True。
        """
        if self.clf is None or self.feature_cols is None:
            return True  # 未训练时不拦截
        # 构造最近特征
        try:
            X, _ = self.build_dataset(df_recent.tail(200))
            if X.empty: return True
            x_last = X.iloc[[-1]][self.feature_cols]
            p = float(self.clf.predict_proba(x_last)[:, 1][0])
            logger.info(f"ML 置信度: {p:.3f} | 阈值: {self.threshold:.3f}")
            return p >= self.threshold
        except Exception as e:
            logger.warning(f"ML 过滤失败（放行）：{e}")
            return True


# ==============================================================================
# 7. 策略 - 重构和优化
# ==============================================================================
class MeanReversionStrategy:
    def __init__(self, events, symbol, z_window, atr_window,
                 z_entry=1.2, z_exit=0.8, atr_multiplier=2.0,  # 更保守的参数
                 cooldown_seconds=600,  # 增加冷却时间到30分钟
                 saved_state=None, hist_feed=None, ml_filter: MLSignalFilter = None):
        self.events = events
        self.symbol = symbol

        self.z_window = z_window
        self.atr_window = atr_window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.atr_multiplier = atr_multiplier
        self.cooldown_seconds = cooldown_seconds

        self.ml_filter = ml_filter

        # 状态
        if saved_state and "strategy" in saved_state:
            self.price_history = deque(saved_state["strategy"]["price_history"], maxlen=z_window + 2)
            self.atr_history = deque(saved_state["strategy"]["atr_history"], maxlen=atr_window + 2)
            self.position = saved_state["strategy"]["position"]
            self.entry_price = saved_state["strategy"]["entry_price"]
            self.last_trade_time = saved_state["strategy"].get("last_trade_time")
            self.consecutive_losses = saved_state["strategy"].get("consecutive_losses", 0)
            self.total_trades = saved_state["strategy"].get("total_trades", 0)
            self.winning_trades = saved_state["strategy"].get("winning_trades", 0)
        else:
            self.price_history = deque(maxlen=z_window + 2)
            self.atr_history = deque(maxlen=atr_window + 2)
            self.position = 0
            self.entry_price = 0.0
            self.last_trade_time = None
            self.consecutive_losses = 0
            self.total_trades = 0
            self.winning_trades = 0

        self.hist_feed = hist_feed
        self._preload_history_data()

    def _preload_history_data(self):
        if len(self.price_history) >= self.z_window:
            return
        try:
            import akshare as ak
            df = ak.fund_etf_hist_sina(symbol=self.symbol, start_date="", end_date="")
            if df.empty:
                logger.warning("历史数据为空，预加载跳过。")
                return

            # 统一列
            close_col = None
            for c in ["close", "收盘"]:
                if c in df.columns: close_col = c
            if close_col is None: close_col = df.columns[-1]
            close = pd.to_numeric(df[close_col], errors="coerce").dropna()
            for price in close.tail(int(max(self.z_window, self.atr_window) * 1.5)).tolist():
                self.price_history.append(price)

            # 构造退化 ATR
            if len(self.price_history) >= 2:
                for i in range(1, len(self.price_history)):
                    tr = abs(self.price_history[i] - self.price_history[i - 1])
                    self.atr_history.append(tr)
            logger.info(f"预加载完成 | 价格: {len(self.price_history)} | ATR: {len(self.atr_history)}")
        except Exception as e:
            logger.warning(f"历史数据预加载失败: {e}")

    def get_state(self):
        return {
            "price_history": list(self.price_history),
            "atr_history": list(self.atr_history),
            "position": self.position,
            "entry_price": self.entry_price,
            "last_trade_time": self.last_trade_time,
            "consecutive_losses": self.consecutive_losses,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades
        }

    def _cooldown_ok(self):
        if not self.last_trade_time: return True
        last = datetime.datetime.strptime(self.last_trade_time, "%Y-%m-%d %H:%M:%S")
        return (datetime.datetime.now() - last).total_seconds() >= self.cooldown_seconds

    def _risk_management_ok(self):
        """增强的风险管理"""
        # 连续亏损次数限制
        if self.consecutive_losses >= 3:
            logger.info("连续亏损3次，暂停交易")
            return False

        # 胜率过低时暂停
        if self.total_trades > 10 and self.winning_trades / self.total_trades < 0.3:
            logger.info(f"胜率过低 {self.winning_trades}/{self.total_trades}，暂停交易")
            return False

        return True

    def calculate_signals(self, event, df_recent_for_ml: pd.DataFrame = None):
        if event.type != 'MARKET' or event.symbol != self.symbol:
            return

        self.price_history.append(event.latest_price)
        if len(self.price_history) >= 2:
            tr = abs(self.price_history[-1] - self.price_history[-2])
            self.atr_history.append(tr)

        if len(self.price_history) < self.z_window:
            logger.debug(f"数据预热中...(价格:{len(self.price_history)}/{self.z_window})")
            return

        prices = pd.Series(list(self.price_history))
        rolling_mean = prices.rolling(window=self.z_window).mean().iloc[-1]
        rolling_std = prices.rolling(window=self.z_window).std().iloc[-1]
        z_score = (event.latest_price - rolling_mean) / (rolling_std if rolling_std > 0 else np.nan)
        if not np.isfinite(z_score): z_score = 0

        atr_val = pd.Series(list(self.atr_history)).rolling(self.atr_window).mean().iloc[-1]
        if not np.isfinite(atr_val): atr_val = 0

        # 计算 RSI 作为额外过滤
        rsi = compute_rsi(prices, 14).iloc[-1]

        logger.info(f"Z={z_score:.2f} | ATR={atr_val:.4f} | RSI={rsi:.1f}")

        # 无仓 -> 超卖买入，但需要多重确认
        if self.position == 0 and self._cooldown_ok() and self._risk_management_ok():
            # 更严格的买入条件：Z-score 超卖 + RSI 超卖 + 价格不在下降趋势
            if (z_score < -self.z_entry and
                    rsi < 30 and  # RSI 超卖
                    prices.iloc[-1] > prices.iloc[-5:].mean()):  # 价格高于近5期均值（避免下降趋势）

                # ML 过滤器
                allow = True
                if self.ml_filter is not None and df_recent_for_ml is not None:
                    allow = self.ml_filter.allow_trade(df_recent_for_ml)
                if allow:
                    logger.info(f"买入信号：Z<{-self.z_entry} & RSI<30 & 价格趋势OK")
                    self.events.put(SignalEvent(event.timestamp, self.symbol, 'LONG'))
                else:
                    logger.info("ML 过滤拒绝本次买入。")
        elif self.position == 1:
            # 更灵活的止盈止损
            stop_loss_price = self.entry_price - self.atr_multiplier * atr_val if atr_val > 0 else self.entry_price * 0.985

            # 动态止盈：根据收益情况调整
            current_return = (event.latest_price / self.entry_price - 1)
            take_profit = False

            if current_return > 0.02:  # 收益 > 2%，更保守止盈
                take_profit = (z_score >= -0.2)  # 几乎回归均值
            elif current_return > 0.01:  # 收益 > 1%
                take_profit = (z_score >= -self.z_exit)
            else:  # 收益较小或亏损，等待更强信号
                take_profit = (z_score >= 0.5) or (rsi > 70)  # 超买信号

            hit_sl = (event.latest_price <= stop_loss_price)

            if take_profit or hit_sl:
                reason = "动态止盈" if take_profit else "ATR 止损"
                logger.info(
                    f"平仓信号（{reason}）| 现价={event.latest_price:.3f} | 止损价={stop_loss_price:.3f} | 收益率={current_return * 100:.2f}%")
                self.events.put(SignalEvent(event.timestamp, self.symbol, 'EXIT'))


# ==============================================================================
# 8. 组合 & 执行 - 修复资金管理
# ==============================================================================
class Portfolio:
    def __init__(self, events, initial_capital, symbol, state_manager=None,
                 max_risk_per_trade=0.015, min_commission=5.0, commission_rate=0.0002354):  # 降低单笔风险
        self.events = events
        self.symbol = symbol
        self.state_manager = state_manager
        self.max_risk_per_trade = max_risk_per_trade
        self.min_commission = min_commission
        self.commission_rate = commission_rate

        saved = state_manager.load_state() if state_manager else None
        if saved and "account" in saved:
            # 验证和修复账户状态
            cash = saved["account"]["cash"]
            position = saved["account"]["position"]

            # 如果现金为负，强制平仓并重置
            if cash < 0:
                logger.warning(f"检测到负现金 {cash:.2f}，强制重置账户")
                self.current_positions = {self.symbol: 0}
                self.current_holdings = {'cash': initial_capital, 'total': initial_capital}
                self.latest_market_value = 0
                self.avg_cost = 0.0
            else:
                self.current_positions = {self.symbol: position}
                self.current_holdings = {'cash': cash, 'total': saved["account"]["total_asset"]}
                self.latest_market_value = saved["account"]["latest_market_value"]
                self.avg_cost = saved["account"].get("avg_cost", 0.0)
        else:
            self.current_positions = {self.symbol: 0}
            self.current_holdings = {'cash': initial_capital, 'total': initial_capital}
            self.latest_market_value = 0
            self.avg_cost = 0.0

    def get_state(self):
        return {
            "cash": self.current_holdings['cash'],
            "total_asset": self.current_holdings['total'],
            "position": self.current_positions[self.symbol],
            "latest_market_value": self.latest_market_value,
            "avg_cost": self.avg_cost
        }

    def _calc_position_size_by_atr(self, price, atr_val):
        """
        改进的头寸计算：更保守且确保不超过可用资金
        """
        if atr_val <= 0 or price <= 0:
            return 0

        # 可用现金（保留5%作为缓冲）
        available_cash = self.current_holdings['cash'] * 0.95
        if available_cash <= 0:
            return 0

        # 基于风险的头寸计算
        stop_dist = max(0.01 * price, 1.5 * atr_val)  # 最小1%止损距离
        max_loss = self.current_holdings['total'] * self.max_risk_per_trade
        risk_based_qty = int(max_loss / stop_dist)

        # 基于资金的头寸限制（最多用70%资金）
        max_affordable = int(available_cash * 0.7 / price)

        # 取较小值
        qty = min(risk_based_qty, max_affordable)

        # 确保有足够资金支付手续费
        total_cost = price * qty + max(price * qty * self.commission_rate, self.min_commission)
        if total_cost > available_cash:
            qty = int((available_cash - self.min_commission) / (price * (1 + self.commission_rate)))

        return max(qty, 0)

    def update_from_fill(self, event, strategy: MeanReversionStrategy):
        if event.type != 'FILL': return

        if event.direction == 'BUY':
            cost = event.fill_price * event.quantity
            total_cost = cost + event.commission

            # 验证资金充足性
            if total_cost > self.current_holdings['cash']:
                logger.error(f"资金不足！需要:{total_cost:.2f}, 可用:{self.current_holdings['cash']:.2f}")
                return

            self.current_holdings['cash'] -= total_cost
            old_position = self.current_positions[event.symbol]

            # 计算平均成本
            if old_position > 0:
                total_old_cost = old_position * self.avg_cost
                self.avg_cost = (total_old_cost + cost) / (old_position + event.quantity)
            else:
                self.avg_cost = event.fill_price

            self.current_positions[event.symbol] = old_position + event.quantity
            strategy.position, strategy.entry_price = 1, self.avg_cost
            strategy.last_trade_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        elif event.direction == 'SELL':
            proceeds = event.fill_price * event.quantity
            net_proceeds = proceeds - event.commission
            self.current_holdings['cash'] += net_proceeds

            # 记录盈亏
            if self.avg_cost > 0:
                profit = (event.fill_price - self.avg_cost) * event.quantity - event.commission
                logger.info(f"交易盈亏: {profit:.2f} RMB")

                # 更新策略统计
                strategy.total_trades += 1
                if profit > 0:
                    strategy.winning_trades += 1
                    strategy.consecutive_losses = 0
                else:
                    strategy.consecutive_losses += 1

            self.current_positions[event.symbol] = 0
            self.avg_cost = 0.0
            strategy.position, strategy.entry_price = 0, 0.0
            strategy.last_trade_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 确保现金不为负
        if self.current_holdings['cash'] < 0:
            logger.error(f"现金变为负数: {self.current_holdings['cash']:.2f}")
            self.current_holdings['cash'] = max(0, self.current_holdings['cash'])

        self.update_portfolio_value()
        if self.state_manager:
            self.state_manager.save_state(self.get_state(), strategy.get_state(), event.transaction_record)

    def on_signal(self, event, latest_price, atr_for_size, strategy: MeanReversionStrategy):
        if event.type != 'SIGNAL' or event.symbol != self.symbol: return

        if event.direction == 'LONG' and self.current_positions[self.symbol] == 0:
            # 动态头寸
            qty = self._calc_position_size_by_atr(latest_price, atr_for_size)
            if qty <= 0:
                logger.info("动态头寸=0，放弃下单。")
                return

            # 最终资金检查
            estimated_cost = qty * latest_price * (1 + self.commission_rate) + self.min_commission
            if estimated_cost > self.current_holdings['cash']:
                logger.warning(f"资金不足，取消下单。需要:{estimated_cost:.2f}, 可用:{self.current_holdings['cash']:.2f}")
                return

            logger.info(f"准备买入 {qty} 股，预计成本 {estimated_cost:.2f} RMB")
            self.events.put(OrderEvent(event.timestamp, self.symbol, 'MKT', qty, 'BUY'))

        elif event.direction == 'EXIT' and self.current_positions[self.symbol] > 0:
            self.events.put(
                OrderEvent(event.timestamp, self.symbol, 'MKT', self.current_positions[self.symbol], 'SELL'))

    def update_portfolio_value(self, latest_price=None):
        if latest_price:
            self.latest_market_value = self.current_positions[self.symbol] * latest_price
        self.current_holdings['total'] = self.current_holdings['cash'] + self.latest_market_value

        # 计算收益率
        unrealized_pnl = 0
        if self.current_positions[self.symbol] > 0 and self.avg_cost > 0 and latest_price:
            unrealized_pnl = (latest_price - self.avg_cost) * self.current_positions[self.symbol]

        logger.info(
            f"账户 | 现金:{self.current_holdings['cash']:.2f} | 持仓市值:{self.latest_market_value:.2f} | 总资产:{self.current_holdings['total']:.2f} | 持仓:{self.current_positions[self.symbol]} | 未实现盈亏:{unrealized_pnl:.2f}")

        if self.state_manager:
            # 注意：此处不写交易记录
            self.state_manager.save_state(self.get_state(), {})


class ExecutionHandler:
    def __init__(self, events, commission_rate, min_commission):
        self.events = events
        self.commission_rate = commission_rate
        self.min_commission = min_commission

    def execute_order(self, event):
        if event.type != 'ORDER':
            return
        fill_price = event.latest_price
        trade_value = fill_price * event.quantity
        commission = max(trade_value * self.commission_rate, self.min_commission)
        self.events.put(
            FillEvent(event.timestamp, event.symbol, event.quantity, event.direction, fill_price, commission))


# ==============================================================================
# 9. 回测与自动调参
# ==============================================================================
class Backtester:
    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy().reset_index(drop=True)
        self.symbol = symbol

    def simulate(self, params, initial_capital=100000, commission_rate=0.0002354, min_commission=5.0):
        """
        仅用于参数评分的快速日线回测（收盘价填充）。
        params: dict 包含 z_window, z_entry, z_exit, atr_window, atr_mult, cooldown, ml_threshold
        """
        close = self.df["close"].astype(float).values
        n = len(close)
        cash, pos, entry = initial_capital, 0, 0.0
        equity = []

        z = rolling_zscore(pd.Series(close), params["z_window"]).values
        atr_series = compute_atr(self.df, params["atr_window"]).values
        rsi_series = compute_rsi(pd.Series(close), 14).values

        last_trade_idx = -10 ** 9
        consecutive_losses = 0
        total_trades = 0
        winning_trades = 0

        for i in range(n):
            price = close[i]
            atrv = atr_series[i]
            rsi_val = rsi_series[i]
            equity.append(cash + pos * price)

            # 持仓期间移动止损
            if pos > 0:
                stop_loss = entry - params["atr_mult"] * atrv if atrv > 0 else entry * 0.985
                current_return = price / entry - 1

                # 动态止盈
                take_profit = False
                if current_return > 0.02:
                    take_profit = z[i] >= -0.2
                elif current_return > 0.01:
                    take_profit = z[i] >= -params["z_exit"]
                else:
                    take_profit = (z[i] >= 0.5) or (rsi_val > 70)

                if price <= stop_loss or take_profit:
                    # 卖出
                    trade_value = price * pos
                    fee = max(trade_value * commission_rate, min_commission)
                    cash += trade_value - fee

                    # 记录交易结果
                    profit = (price - entry) * pos - fee - max(entry * pos * commission_rate, min_commission)
                    total_trades += 1
                    if profit > 0:
                        winning_trades += 1
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1

                    pos = 0
                    last_trade_idx = i
                    entry = 0.0
                    continue

            # 风险管理检查
            risk_ok = (consecutive_losses < 3 and
                       (total_trades <= 10 or winning_trades / total_trades >= 0.3))

            # 开仓 - 更严格条件
            if (pos == 0 and risk_ok and
                    z[i] < -params["z_entry"] and
                    rsi_val < 30 and
                    i >= 5 and price > np.mean(close[i - 4:i + 1])):  # 价格趋势检查

                if i - last_trade_idx < int(params["cooldown"] / 1440):  # 日线冷却转换
                    continue

                # 动态头寸
                stop_dist = max(0.01 * price, 1.5 * atrv if atrv > 0 else 0.01 * price)
                max_loss = initial_capital * 0.008
                qty = int(max_loss / stop_dist)

                # 资金限制
                available_cash = cash * 0.7
                max_affordable = int(available_cash / price)
                qty = min(qty, max_affordable)

                if qty <= 0:
                    continue

                trade_value = price * qty
                fee = max(trade_value * commission_rate, min_commission)
                total_cost = trade_value + fee

                if total_cost > cash:
                    continue

                cash -= total_cost
                pos = qty
                entry = price
                last_trade_idx = i

        equity = pd.Series(equity)
        rets = equity.pct_change().fillna(0)
        sr = sharpe_ratio(rets)
        mdd = max_drawdown(equity)
        return {"sharpe": sr, "mdd": mdd, "equity": equity, "total_trades": total_trades,
                "win_rate": winning_trades / max(total_trades, 1)}

    def random_search(self, n_iter=50, lam=3.0, seed=42):  # 增加迭代次数和lambda
        random.seed(seed)
        best_score, best_params, best_metrics = -1e9, None, None
        for k in range(n_iter):
            params = {
                "z_window": random.choice([20, 30, 40]),
                "z_entry": random.choice([1.2, 1.5, 1.8, 2.0]),  # 加入更低的入场值
                "z_exit": random.choice([0.5, 0.8, 1.0]),  # 加入更高的出场值
                "atr_window": random.choice([14, 20]),
                "atr_mult": random.choice([1.8, 2.0, 2.5]),  # 加入更宽的止损
                "cooldown": random.choice([600, 1200, 1800]),  # 加入更短的冷却
                "ml_threshold": random.choice([0.55, 0.60, 0.65])  # 放宽ML
            }

            metrics = self.simulate(params)
            # 综合评分：考虑夏普、回撤、胜率、交易频率
            score = (metrics["sharpe"] -
                     lam * abs(metrics["mdd"]) +
                     metrics["win_rate"] * 2 -
                     abs(metrics["total_trades"] - 50) * 0.01)  # 偏好适中交易频率

            logger.info(
                f"[调参] {k + 1:02d}/{n_iter} | Sharpe={metrics['sharpe']:.3f} | MDD={metrics['mdd']:.3f} | 胜率={metrics['win_rate']:.3f} | 交易={metrics['total_trades']} | Score={score:.3f}")
            if score > best_score:
                best_score, best_params, best_metrics = score, params, metrics

        logger.info(f"[调参] 最优参数: {best_params}")
        logger.info(
            f"[调参] 最优指标: Sharpe={best_metrics['sharpe']:.3f} | MDD={best_metrics['mdd']:.3f} | 胜率={best_metrics['win_rate']:.3f} | 交易次数={best_metrics['total_trades']}")
        return best_params, best_metrics


# ==============================================================================
# 10. 主程序
# ==============================================================================

# 创建一个全局的数据库管理器实例
db_manager = DBManager()


def fetch_hist(symbol: str, days: int = 600) -> pd.DataFrame:
    """
    智能拉取历史 K 线：优先读库，只从网络获取增量数据。
    返回标准列：date, open, high, low, close, volume
    """
    import akshare as ak

    # 1. 从本地数据库读取现有数据
    df_local = db_manager.read_data(symbol)

    last_date = None
    if not df_local.empty:
        last_date = df_local['date'].max()

    # 2. 检查是否需要更新数据
    # 如果没有数据，或者最后一条数据是昨天之前的，则需要更新
    needs_update = True
    if last_date:
        # 如果最后日期是今天或昨天（考虑到休市），可能不需要更新
        if last_date.date() >= (datetime.date.today() - datetime.timedelta(days=1)):
            needs_update = False
            logger.info("本地数据已是最新，无需从网络获取。")

    if not needs_update:
        return df_local.tail(days).reset_index(drop=True)

    # 3. 从 akshare 获取全量数据 (因为其接口不支持日期范围)
    logger.info("正在从 akshare 获取最新数据...")
    try:
        df_remote = ak.fund_etf_hist_sina(symbol=symbol)
        if df_remote is None or df_remote.empty:
            logger.warning(f"从 akshare 获取 {symbol} 数据失败，返回本地数据。")
            return df_local.tail(days).reset_index(drop=True)
    except Exception as e:
        logger.error(f"调用 akshare 接口异常: {e}，返回本地数据。")
        return df_local.tail(days).reset_index(drop=True)

    # 4. 数据清洗与格式化 (与原代码类似)
    colmap = {}
    for c in df_remote.columns:
        lc = c.lower()
        if "date" in lc or "日期" in c:
            colmap[c] = "date"
        elif lc in ("open", "开盘"):
            colmap[c] = "open"
        elif lc in ("high", "最高"):
            colmap[c] = "high"
        elif lc in ("low", "最低"):
            colmap[c] = "low"
        elif lc in ("close", "收盘"):
            colmap[c] = "close"
        elif "volume" in lc or "成交量" in c:
            colmap[c] = "volume"
    df_remote = df_remote.rename(columns=colmap)
    df_remote["date"] = pd.to_datetime(df_remote["date"], errors="coerce")
    df_remote = df_remote.dropna(subset=['date'])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df_remote.columns:
            df_remote[c] = pd.to_numeric(df_remote[c], errors="coerce")
        else:  # 补充缺失列
            df_remote[c] = df_remote['close'] if 'close' in df_remote.columns else np.nan

    # 5. 筛选出需要写入数据库的新数据
    df_new = df_remote
    if last_date:
        df_new = df_remote[df_remote['date'] > last_date]

    # 6. 将新数据写入数据库
    if not df_new.empty:
        db_manager.write_data(df_new, symbol)

    # 7. 合并本地和新数据，返回最终结果
    df_final = pd.concat([df_local, df_new], ignore_index=True).drop_duplicates(subset=['date'], keep='last')
    df_final = df_final.sort_values("date").reset_index(drop=True)

    required_cols = ["date", "open", "high", "low", "close", "volume"]
    return df_final[required_cols].tail(days).reset_index(drop=True)

def main():
    # ===== 用户可配置区域 =====
    SYMBOL = '159509.SZ'
    INITIAL_CAPITAL = 100000.0
    COMMISSION_RATE = 0.0002354
    MIN_COMMISSION = 5.0
    MAX_RISK_PER_TRADE = 0.008  # 降低单笔风险到0.8%

    # 若首次运行，建议先允许自动调参和 ML 训练
    ENABLE_AUTO_TUNE = True
    ENABLE_ML = True
    ML_MODEL_PATH = f"ml_filter_{SYMBOL.replace('.', '_')}.joblib"

    # ===== 初始化 =====
    state_manager = StateManager(SYMBOL)
    saved_state = state_manager.load_state()

    # 历史数据（回测/训练）
    df_hist = fetch_hist(SYMBOL, days=800)
    if df_hist.empty:
        logger.warning("无法获取历史数据，ML/调参将被跳过。")

    # 自动调参 - 使用更保守的默认参数
    default_params = dict(z_window=40, z_entry=1.8, z_exit=0.3, atr_window=20, atr_mult=1.5, cooldown=1800,
                          ml_threshold=0.65)
    tuned_params = default_params

    if ENABLE_AUTO_TUNE and not df_hist.empty:
        tuner = Backtester(df_hist, SYMBOL)
        tuned_params, metrics = tuner.random_search(n_iter=50, lam=3.0, seed=42)
        logger.info(f"采用调参结果: {tuned_params}")

    # 训练 ML
    ml_filter = None
    if ENABLE_ML and not df_hist.empty:
        ml_filter = MLSignalFilter(model_path=ML_MODEL_PATH, n_days_take=5, ret_target=0.008, risk_cap=0.03)  # 更保守目标
        if not ml_filter.load():
            ml_filter.train(df_hist, threshold=tuned_params.get("ml_threshold", 0.65))
        else:
            # 同步阈值
            ml_filter.threshold = tuned_params.get("ml_threshold", ml_filter.threshold)

    # 组件
    events = queue.Queue()
    bars = MultiSourceDataHandler(events, SYMBOL)
    strategy = MeanReversionStrategy(
        events, SYMBOL,
        z_window=tuned_params["z_window"],
        atr_window=tuned_params["atr_window"],
        z_entry=tuned_params["z_entry"],
        z_exit=tuned_params["z_exit"],
        atr_multiplier=tuned_params["atr_mult"],
        cooldown_seconds=tuned_params["cooldown"],
        saved_state=saved_state,
        hist_feed=df_hist,
        ml_filter=ml_filter
    )
    port = Portfolio(events, INITIAL_CAPITAL, SYMBOL, state_manager,
                     max_risk_per_trade=MAX_RISK_PER_TRADE,
                     min_commission=MIN_COMMISSION,
                     commission_rate=COMMISSION_RATE)
    broker = ExecutionHandler(events, COMMISSION_RATE, MIN_COMMISSION)

    logger.info("=" * 70)
    logger.info("优化版 ETF 交易机器人已启动")
    logger.info(f"标的: {SYMBOL}")
    logger.info(
        f"参数: Z入场={tuned_params['z_entry']} | Z出场={tuned_params['z_exit']} | Z窗口={tuned_params['z_window']}")
    logger.info(
        f"     ATR窗口={tuned_params['atr_window']} | ATR倍数={tuned_params['atr_mult']} | 冷却={tuned_params['cooldown']}s")
    logger.info(f"     ML阈值={tuned_params.get('ml_threshold', 0.65)} | 单笔风险={MAX_RISK_PER_TRADE * 100:.1f}%")
    logger.info("=" * 70)

    latest_price = 0.0
    try:
        while True:
            if is_market_open_cn():
                data_ok = bars.update_bars()
                while not events.empty():
                    try:
                        ev = events.get(block=False)
                    except queue.Empty:
                        break
                    else:
                        if ev.type == 'MARKET':
                            latest_price = ev.latest_price
                            # 供 ML 过滤：将最新价更新到 df_hist，保持最近窗口
                            # 供 ML 过滤：将最新价更新到 df_hist，保持最近窗口
                            if not df_hist.empty:
                                new_row = {
                                    "date": pd.Timestamp(datetime.datetime.now()),
                                    "open": latest_price,
                                    "high": latest_price,
                                    "low": latest_price,
                                    "close": latest_price,
                                    "volume": 0
                                }
                                df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
                                # 保持最近 500 条记录
                                if len(df_hist) > 500:
                                    df_hist = df_hist.iloc[-500:].reset_index(drop=True)

                            port.update_portfolio_value(latest_price)
                            atr_for_size = \
                            pd.Series(list(strategy.atr_history)).rolling(strategy.atr_window).mean().iloc[-1] if len(
                                strategy.atr_history) >= strategy.atr_window else 0.01
                            strategy.calculate_signals(ev, df_hist)

                        elif ev.type == 'SIGNAL':
                            port.on_signal(ev, latest_price, atr_for_size, strategy)

                        elif ev.type == 'ORDER':
                            ev.latest_price = latest_price
                            broker.execute_order(ev)

                        elif ev.type == 'FILL':
                            port.update_from_fill(ev, strategy)

                if data_ok:
                    time.sleep(bars.get_random_interval())
                else:
                    time.sleep(60)  # 数据获取失败时较长休眠
            else:
                logger.info("非交易时间，等待中...")
                time.sleep(300)  # 非交易时间每5分钟检查一次

    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在保存状态并退出...")
    except Exception as e:
        logger.exception(f"程序异常: {e}")
    finally:
        # 保存最终状态
        if 'strategy' in locals() and 'port' in locals():
            state_manager.save_state(port.get_state(), strategy.get_state())
        logger.info("程序已安全退出")
        gc.collect()


if __name__ == "__main__":
    main()