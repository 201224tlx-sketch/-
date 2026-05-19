import pandas as pd
import numpy as np
import os
import random
import requests
import time
from fake_useragent import UserAgent
import akshare as ak
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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


class StockRegressionAnalyzer:
    def __init__(self, symbol='601869.SH'):
        self.symbol = symbol
        self.symbol_code = symbol.split('.')[0]
        self.model = LinearRegression()
        self.features = ['open', 'high', 'low', 'volume', 'ma5', 'ma10']
        self.latest_daily_data = None

    def fetch_data(self):
        print(f"\n[{self.symbol}] 正在获取历史日线数据...")
        global requests
        requests.get = original_get

        df = pd.DataFrame()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                df = ak.stock_zh_a_hist(symbol=self.symbol_code, period="daily", start_date="20200101", adjust="qfq")
                col_map = {
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'
                }
                df.rename(columns=col_map, inplace=True)
                print(f"成功通过主数据源(东方财富)获取数据，共 {len(df)} 条记录。")
                break
            except Exception as e:
                print(f"主数据源尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("尝试备用数据源(新浪)...")
                    try:
                        sina_symbol = f"sh{self.symbol_code}" if "SH" in self.symbol else f"sz{self.symbol_code}"
                        df = ak.stock_zh_a_daily(symbol=sina_symbol, adjust="qfq")
                        df.reset_index(inplace=True)
                        if 'date' not in df.columns:
                            df.rename(columns={df.columns[0]: 'date'}, inplace=True)
                        print(f"成功通过备用数据源(新浪)获取数据，共 {len(df)} 条记录。")
                    except Exception as e2:
                        print(f"备用数据源获取失败: {e2}")

        requests.get = ultra_safe_request

        if df.empty:
            print("未能获取到有效数据。")
            return None

        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'close', 'high', 'low', 'volume']
        if 'amount' in df.columns: numeric_cols.append('amount')

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def train_and_evaluate(self, df):
        print("\n=== 开始特征工程与回归模型训练 ===")
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()

        self.features = ['open', 'high', 'low', 'volume', 'ma5', 'ma10']
        if 'amount' in df.columns:
            self.features.append('amount')

        self.latest_daily_data = df.iloc[-1].copy()
        df['target_next_close'] = df['close'].shift(-1)
        df_clean = df.dropna(subset=self.features + ['target_next_close']).copy()

        X = df_clean[self.features]
        y = df_clean['target_next_close']

        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)

        coef_dict = dict(zip(self.features, [round(c, 5) for c in self.model.coef_]))
        print(f"模型训练完成! 训练样本数: {len(df_clean)} 天")
        print(f"整体拟合优度 (R²): {r2:.4f}")
        print("各解释变量对次日股价的影响权重 (偏回归系数):")
        for k, v in coef_dict.items():
            print(f" - {k.upper()}: {v}")

        plot_df = df_clean.tail(100)
        dates_test = plot_df['date']
        y_test = plot_df['target_next_close']
        y_pred_test = self.model.predict(plot_df[self.features])
        return dates_test, y_test, y_pred_test, coef_dict

    def predict_future(self):
        print("\n" + "=" * 20 + "未来股价走势预测" + "=" * 20)
        if self.latest_daily_data is None:
            return

        try:
            latest_date = self.latest_daily_data['date']
            date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else \
            str(latest_date).split(' ')[0]
            latest_close = float(self.latest_daily_data['close'])

            X_latest = pd.DataFrame([self.latest_daily_data[self.features]])
            predicted_close = self.model.predict(X_latest)[0]
            price_diff = predicted_close - latest_close
            pct_change = (price_diff / latest_close) * 100

            if price_diff > 0:
                trend = "上涨 📈"
            elif price_diff < 0:
                trend = "下跌 📉"
            else:
                trend = "平盘 ➖"

            print(f"1. 基准日期 (最新交易日): {date_str}")
            print(f"2. 当日实际收盘价: {latest_close:.2f} 元")
            print(f"3. 模型预测下一交易日收盘价: {predicted_close:.2f} 元")
            print(f"4. 预期次日走势: {trend} (预期涨跌幅: {pct_change:.2f}%)")
            print("=" * 61)
        except Exception as e:
            print(f"预测未来价格时发生异常: {e}")

    def plot_results(self, dates_test, y_test, y_pred, coef_dict):
        print("\n正在生成可视化图表...")
        fig = plt.figure(figsize=(15, 10))

        ax1 = fig.add_subplot(2, 1, 1)
        features = list(coef_dict.keys())
        coefs = list(coef_dict.values())
        colors = ['#ff5252' if c > 0 else '#4caf50' for c in coefs]
        bars = ax1.bar(features, coefs, color=colors, alpha=0.8)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_title(f'[{self.symbol}] 股价多元回归特征权重分析', fontsize=14)
        ax1.set_ylabel('偏回归系数大小')
        ax1.grid(axis='y', linestyle='--', alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            offset = 10 if yval > 0 else -15
            ax1.annotate(f'{yval:.5f}', xy=(bar.get_x() + bar.get_width() / 2, yval), xytext=(0, offset),
                         textcoords="offset points", ha='center', va='bottom' if yval > 0 else 'top', fontsize=10)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(dates_test, y_test, label='真实次日收盘价', color='#1f77b4', linewidth=2, marker='.')
        ax2.plot(dates_test, y_pred, label='模型预测收盘价', color='#ff7f0e', linestyle='--', linewidth=2)
        ax2.set_title(f'模型拟合效果对比 (最近100个交易日)', fontsize=14)
        ax2.set_ylabel('股票价格 (元)')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path = f"{self.symbol_code}_regression_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {os.path.abspath(save_path)}")
        plt.show()


if __name__ == '__main__':
    TARGET_SYMBOL = '601869.SH'
    analyzer = StockRegressionAnalyzer(symbol=TARGET_SYMBOL)
    df_data = analyzer.fetch_data()

    if df_data is not None:
        dates_test, y_test, y_pred, coef_dict = analyzer.train_and_evaluate(df_data)
        analyzer.predict_future()
        analyzer.plot_results(dates_test, y_test, y_pred, coef_dict)

    print("\n程序运行结束。")