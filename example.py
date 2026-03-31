import queue
import time
import datetime
from collections import deque
import pandas as pd
import numpy as np
import os
import random
import requests
import pickle
from fake_useragent import UserAgent

# ==============================================================================
# 1. 反爬优化配置
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
# 2. 长期记忆功能
# ==============================================================================
class StateManager:
    def __init__(self, symbol, state_file="trading_state.pkl"):
        self.state_file = state_file
        self.symbol = symbol
        self.default_state = {
            "account": {
                "cash": 100000.0,
                "total_asset": 100000.0,
                "position": 0,  # 1=多仓, -1=空仓, 0=空仓
                "latest_market_value": 0.0
            },
            "strategy": {
                "price_history": [],
                "atr_history": [],
                "position": 0,
                "entry_price": 0.0
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
                    print(f"成功加载历史状态，上次更新时间: {state['system']['last_update_time']}")
                    print(f"已运行次数: {state['system']['run_count'] + 1}次")
                    print(f"当前持仓: {state['account']['position']}股")
                    print(f"当前总资产: {state['account']['total_asset']:.2f}元")
                    return state
                else:
                    print("状态文件标的不匹配，使用新状态")
                    return self._init_new_state()
            except Exception as e:
                print(f"加载状态失败: {e}，使用新状态")
                return self._init_new_state()
        else:
            print("未找到历史状态，初始化新状态")
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
                        [transaction] if transaction else [])
                }
            }

            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
            return True
        except Exception as e:
            print(f"保存状态失败: {e}")
            return False


# ==============================================================================
# 3. 多源数据处理器
# ==============================================================================
class MultiSourceDataHandler:
    def __init__(self, events, symbol):
        self.events = events
        self.symbol = symbol
        self.symbol_no_suffix = symbol.replace(".SZ", "").replace(".SH", "")
        self.failure_counts = {
            "akshare_em": 0,
            "akshare_sina": 0,
            "tencent": 0
        }
        self.max_fail = 3
        self.cooldowns = {}
        self.base_interval = 30

    def _is_source_available(self, source):
        return time.time() > self.cooldowns.get(source, 0)

    def _fetch_akshare_em(self):
        if not self._is_source_available("akshare_em"):
            return None
        try:
            df = ak.fund_etf_spot_em()
            row = df[df["代码"] == self.symbol_no_suffix]
            if not row.empty and pd.notna(row["最新价"].iloc[0]):
                self.failure_counts["akshare_em"] = 0
                return {
                    "price": float(row["最新价"].iloc[0]),
                    "source": "东方财富(em)"
                }
        except Exception as e:
            self.failure_counts["akshare_em"] += 1
            if self.failure_counts["akshare_em"] >= self.max_fail:
                self.cooldowns["akshare_em"] = time.time() + 300
                print(f"\n | 东方财富(em)连续失败，冷却5分钟 | 错误: {str(e)[:30]}...")
        return None

    def _fetch_akshare_sina(self):
        if not self._is_source_available("akshare_sina"):
            return None
        try:
            df = ak.fund_etf_spot_sina(symbol=self.symbol)
            if not df.empty and "现价" in df.columns and pd.notna(df["现价"].iloc[0]):
                self.failure_counts["akshare_sina"] = 0
                return {
                    "price": float(df["现价"].iloc[0]),
                    "source": "新浪财经(sina)"
                }
        except Exception as e:
            self.failure_counts["akshare_sina"] += 1
            if self.failure_counts["akshare_sina"] >= self.max_fail:
                self.cooldowns["akshare_sina"] = time.time() + 300
                print(f"\n | 新浪财经(sina)连续失败，冷却5分钟 | 错误: {str(e)[:30]}...")
        return None

    def _fetch_tencent(self):
        if not self._is_source_available("tencent"):
            return None
        try:
            market = "sz" if ".SZ" in self.symbol else "sh"
            url = f"https://qt.gtimg.cn/q={market}{self.symbol_no_suffix}"
            response = requests.get(url, headers=get_super_headers(), timeout=10)
            data = response.text.split("~")
            if len(data) > 4 and data[3] != "" and data[3].replace('.', '', 1).isdigit():
                price = float(data[3])
                if price > 0:
                    self.failure_counts["tencent"] = 0
                    return {
                        "price": price,
                        "source": "腾讯财经(gtimg)"
                    }
        except Exception as e:
            self.failure_counts["tencent"] += 1
            if self.failure_counts["tencent"] >= self.max_fail:
                self.cooldowns["tencent"] = time.time() + 300
                print(f"\n | 腾讯财经连续失败，冷却5分钟 | 错误: {str(e)[:30]}...")
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
                print(f"最新价: {data['price']:.3f} (来源: {data['source']})")
                self.events.put(MarketEvent(datetime.datetime.now(), self.symbol, data['price']))
                return True

        sleep_time = random.randint(60, 180)
        print(f"\n | 所有数据源失败，休眠{sleep_time}秒...")
        time.sleep(sleep_time)
        return False

    def get_random_interval(self):
        return self.base_interval + random.uniform(-10, 10)


# ==============================================================================
# 4. 事件定义
# ==============================================================================
class Event: pass


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
        print(
            f"\n | 成交: {self.direction} {self.quantity} 股 {self.symbol} @ {self.fill_price:.3f}, 手续费: {self.commission:.2f} RMB")

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
# 5. 策略模块（核心优化部分）
# ==============================================================================
class MeanReversionStrategy:
    def __init__(self, events, z_window, atr_window, z_threshold, atr_multiplier, saved_state=None):
        self.events = events
        self.symbol = SYMBOL
        self.z_window = z_window  # Z-Score计算窗口
        self.atr_window = atr_window  # ATR计算窗口
        self.z_threshold = z_threshold  # Z-Score阈值（降低至1.0）
        self.atr_multiplier = atr_multiplier  # ATR止损倍数

        # 从保存的状态加载或初始化
        if saved_state and "strategy" in saved_state:
            self.price_history = deque(saved_state["strategy"]["price_history"], maxlen=z_window + 1)
            self.atr_history = deque(saved_state["strategy"]["atr_history"], maxlen=atr_window)
            self.position = saved_state["strategy"]["position"]  # 1=多仓, -1=空仓, 0=空仓
            self.entry_price = saved_state["strategy"]["entry_price"]
        else:
            self.price_history = deque(maxlen=z_window + 1)
            self.atr_history = deque(maxlen=atr_window)
            self.position = 0
            self.entry_price = 0.0

        self._preload_history_data()

    def _preload_history_data(self):
        """预加载历史数据，兼容不同版本的列名"""
        if len(self.price_history) < self.z_window:
            print("正在预加载历史数据...")
            try:
                df = ak.fund_etf_hist_sina(symbol=self.symbol, start_date="", end_date="")
                if not df.empty:
                    # 兼容不同版本的列名（"close"或"收盘"）
                    if "close" in df.columns:
                        close_col = "close"
                    elif "收盘" in df.columns:
                        close_col = "收盘"
                    else:
                        close_col = df.columns[-1]  # 最后一列通常是收盘价
                    print(f"检测到收盘价列: {close_col}")

                    df[close_col] = pd.to_numeric(df[close_col], errors='coerce').dropna()
                    max_needed = max(self.z_window, self.atr_window)
                    history_prices = df[close_col].tail(int(max_needed * 1.5)).tolist()

                    # 填充价格历史
                    for price in history_prices:
                        self.price_history.append(price)

                    # 预计算ATR历史
                    if len(self.price_history) >= 2:
                        for i in range(1, len(self.price_history)):
                            tr = abs(self.price_history[i] - self.price_history[i - 1])
                            self.atr_history.append(tr)

                    print(f"预加载完成，价格数据: {len(self.price_history)}条，ATR数据: {len(self.atr_history)}条")
            except Exception as e:
                print(f"历史数据预加载失败: {e}")

    def get_state(self):
        return {
            "price_history": list(self.price_history),
            "atr_history": list(self.atr_history),
            "position": self.position,
            "entry_price": self.entry_price
        }

    def calculate_signals(self, event):
        if event.type == 'MARKET' and event.symbol == self.symbol:
            # 更新价格历史
            self.price_history.append(event.latest_price)

            # 计算并更新ATR（真正的ATR计算）
            if len(self.price_history) >= 2:
                tr = abs(self.price_history[-1] - self.price_history[-2])  # 真实波动幅度
                self.atr_history.append(tr)

            # 检查是否满足计算条件
            if len(self.price_history) < self.z_window:
                print(f"数据预热中...(价格数据: {len(self.price_history)}/{self.z_window})")
                return

            # 计算Z-Score（核心指标）
            prices = pd.Series(list(self.price_history))
            rolling_mean = prices.rolling(window=self.z_window).mean().iloc[-1]
            rolling_std = prices.rolling(window=self.z_window).std().iloc[-1]
            z_score = (event.latest_price - rolling_mean) / rolling_std if rolling_std > 0 else 0

            # 计算当前ATR值
            atr = pd.Series(list(self.atr_history)).mean() if len(self.atr_history) >= self.atr_window else 0
            print(f"Z-Score: {z_score:.2f} | ATR: {atr:.4f} (窗口: {len(self.atr_history)}/{self.atr_window})")

            # ==================================================================
            # 交易信号逻辑（根据建议优化）
            # ==================================================================
            # 方法一：放宽买入条件 - 只要Z-Score超卖就买
            if self.position == 0:
                # 超卖信号（低于负阈值）
                if z_score < -self.z_threshold:
                    print(f"买入信号: Z-Score超卖 ({z_score:.2f} < -{self.z_threshold})")
                    self.events.put(SignalEvent(event.timestamp, self.symbol, 'LONG'))
                # 超买信号（高于正阈值）- 可扩展做空逻辑
                elif z_score > self.z_threshold:
                    print(f"卖出信号: Z-Score超买 ({z_score:.2f} > {self.z_threshold})")
                    # self.events.put(SignalEvent(event.timestamp, self.symbol, 'SHORT'))  # 如需做空取消注释

            # 持仓状态下的退出逻辑
            elif self.position == 1:  # 多仓
                # 计算动态止损价（基于ATR）
                stop_loss_price = self.entry_price - self.atr_multiplier * atr if atr > 0 else self.entry_price * 0.98
                # 回归均值止盈或止损退出
                if event.latest_price >= rolling_mean or event.latest_price <= stop_loss_price:
                    reason = "止盈 (回归均值)" if event.latest_price >= rolling_mean else f"止损 (ATR={atr:.4f})"
                    print(f"平仓信号 ({reason}): 现价={event.latest_price:.3f}, 止损价={stop_loss_price:.3f}")
                    self.events.put(SignalEvent(event.timestamp, self.symbol, 'EXIT'))

            # 如需支持做空，可添加空头逻辑
            # elif self.position == -1:  # 空仓
            #     # 空头止损和止盈逻辑
            #     pass


# ==============================================================================
# 6. 投资组合与执行模块
# ==============================================================================
class Portfolio:
    def __init__(self, events, initial_capital, position_size, saved_state=None, state_manager=None):
        self.events = events
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.symbol = SYMBOL
        self.state_manager = state_manager

        # 从保存的状态加载或初始化
        if saved_state and "account" in saved_state:
            self.current_positions = {self.symbol: saved_state["account"]["position"]}
            self.current_holdings = {
                'cash': saved_state["account"]["cash"],
                'total': saved_state["account"]["total_asset"]
            }
            self.latest_market_value = saved_state["account"]["latest_market_value"]
        else:
            self.current_positions = {self.symbol: 0}
            self.current_holdings = {'cash': initial_capital, 'total': initial_capital}
            self.latest_market_value = 0

    def get_state(self):
        return {
            "cash": self.current_holdings['cash'],
            "total_asset": self.current_holdings['total'],
            "position": self.current_positions[self.symbol],
            "latest_market_value": self.latest_market_value
        }

    def update_from_fill(self, event):
        if event.type == 'FILL':
            if event.direction == 'BUY':
                cost = event.fill_price * event.quantity
                self.current_holdings['cash'] -= (cost + event.commission)
                self.current_positions[event.symbol] = event.quantity  # 多仓
                strategy.position, strategy.entry_price = 1, event.fill_price
            elif event.direction == 'SELL':
                proceeds = event.fill_price * event.quantity
                self.current_holdings['cash'] += (proceeds - event.commission)
                self.current_positions[event.symbol] = 0  # 平仓
                strategy.position, strategy.entry_price = 0, 0
            # 如需支持做空，添加空头处理逻辑
            # elif event.direction == 'SHORT':
            #     ...

            self.update_portfolio_value()
            if self.state_manager:
                self.state_manager.save_state(
                    self.get_state(),
                    strategy.get_state(),
                    event.transaction_record
                )

    def on_signal(self, event):
        if event.type == 'SIGNAL' and event.symbol == self.symbol:
            if event.direction == 'LONG' and self.current_positions[self.symbol] == 0:
                self.events.put(OrderEvent(event.timestamp, self.symbol, 'MKT', self.position_size, 'BUY'))
            elif event.direction == 'EXIT' and self.current_positions[self.symbol] > 0:
                self.events.put(
                    OrderEvent(event.timestamp, self.symbol, 'MKT', self.current_positions[self.symbol], 'SELL'))
            # 如需支持做空，添加空头订单逻辑

    def update_portfolio_value(self, latest_price=None):
        if latest_price:
            self.latest_market_value = self.current_positions[self.symbol] * latest_price
        self.current_holdings['total'] = self.current_holdings['cash'] + self.latest_market_value
        print(
            f"账户状态 | 现金: {self.current_holdings['cash']:.2f}, 持仓市值: {self.latest_market_value:.2f}, 总资产: {self.current_holdings['total']:.2f}, 持仓: {self.current_positions[self.symbol]} 股")

        if self.state_manager:
            self.state_manager.save_state(self.get_state(), strategy.get_state())


class ExecutionHandler:
    def __init__(self, events, commission_rate, min_commission):
        self.events = events
        self.commission_rate = commission_rate
        self.min_commission = min_commission

    def execute_order(self, event):
        if event.type == 'ORDER':
            fill_price = event.latest_price
            trade_value = fill_price * event.quantity
            commission = max(trade_value * self.commission_rate, self.min_commission)
            self.events.put(
                FillEvent(event.timestamp, event.symbol, event.quantity, event.direction, fill_price, commission))


# ==============================================================================
# 7. 辅助函数
# ==============================================================================
def is_market_open():
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    morning_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    morning_close = now.replace(hour=11, minute=30, second=0, microsecond=0)
    afternoon_open = now.replace(hour=13, minute=0, second=0, microsecond=0)
    afternoon_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return (morning_open <= now <= morning_close) or (afternoon_open <= now <= afternoon_close)


# ==============================================================================
# 8. 主程序（优化参数）
# ==============================================================================
if __name__ == '__main__':
    import akshare as ak

    SYMBOL = '159509.SZ'
    INITIAL_CAPITAL = 100000.0
    COMMISSION_RATE = 0.0002354
    MIN_COMMISSION = 5.0
    POSITION_SIZE = 1000

    # 策略参数（根据建议优化）
    Z_SCORE_WINDOW = 30  # Z-Score计算窗口
    Z_SCORE_THRESHOLD = 1.0  # 降低阈值，更易触发（原为2.0）
    ATR_WINDOW = 14  # ATR计算窗口
    ATR_MULTIPLIER = 2.5  # ATR止损倍数

    state_manager = StateManager(SYMBOL)
    saved_state = state_manager.load_state()

    events = queue.Queue()
    bars = MultiSourceDataHandler(events, SYMBOL)
    # 注意：移除了长期均线参数，简化策略
    strategy = MeanReversionStrategy(events, Z_SCORE_WINDOW, ATR_WINDOW,
                                     Z_SCORE_THRESHOLD, ATR_MULTIPLIER, saved_state)
    port = Portfolio(events, INITIAL_CAPITAL, POSITION_SIZE, saved_state, state_manager)
    broker = ExecutionHandler(events, COMMISSION_RATE, MIN_COMMISSION)

    print("=" * 50)
    print("优化后的ETF交易机器人已启动")
    print(f"监控标的: {SYMBOL} | Z-Score阈值: {Z_SCORE_THRESHOLD}")
    print("=" * 50)

    latest_market_price = 0.0

    try:
        while True:
            if is_market_open():
                data_updated = bars.update_bars()

                while not events.empty():
                    try:
                        event = events.get(block=False)
                    except queue.Empty:
                        break
                    else:
                        if event.type == 'MARKET':
                            latest_market_price = event.latest_price
                            strategy.calculate_signals(event)
                            port.update_portfolio_value(event.latest_price)
                        elif event.type == 'SIGNAL':
                            port.on_signal(event)
                        elif event.type == 'ORDER':
                            event.latest_price = latest_market_price
                            broker.execute_order(event)
                        elif event.type == 'FILL':
                            port.update_from_fill(event)

                sleep_time = bars.get_random_interval()
                print(f"等待 {sleep_time:.1f} 秒后再次请求...")
                time.sleep(sleep_time)
            else:
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                off_hours_sleep = 300
                print(f"\r市场已关闭，{off_hours_sleep}秒后再次检查... {current_time}", end="")
                time.sleep(off_hours_sleep)
    except KeyboardInterrupt:
        print("\n程序被手动中断，保存当前状态...")
        state_manager.save_state(port.get_state(), strategy.get_state())
    finally:
        print("\n程序已退出")
