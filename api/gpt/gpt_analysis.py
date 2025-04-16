import json
import pandas as pd
import requests
import openai
from datetime import datetime
from api.config.application import GPT_API_KEY
import logging
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import time
from typing import Dict, List

client = openai.AsyncClient(api_key=GPT_API_KEY)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

TIMEFRAMES = {
    "5m": {"interval": "5m", "name": "5 Minutes", "weight": 0.1},
    "15m": {"interval": "15m", "name": "15 Minutes", "weight": 0.15},
    "1h": {"interval": "1h", "name": "1 Hour", "weight": 0.25},
    "4h": {"interval": "4h", "name": "4 Hours", "weight": 0.25},
    "1d": {"interval": "1d", "name": "Daily", "weight": 0.25}
}


class RateLimiter:
    """Request rate limiter"""

    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            self.requests = [req for req in self.requests if now - req < self.time_window]
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.requests.append(now)


def fetch_data(url: str, params: dict = None, retries: int = 3) -> dict:
    """General data fetcher with retry mechanism"""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)  # 10 seconds timeout
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logging.error(f"Final request failed: {url}")
                return None


def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> tuple:
    """Calculate support and resistance levels"""
    rolling_low = df['low'].rolling(window=window).min()
    rolling_high = df['high'].rolling(window=window).max()
    support = rolling_low.iloc[-1]
    resistance = rolling_high.iloc[-1]
    return support, resistance


def get_binance_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    """Fetch Binance kline (candlestick) data"""
    url = f"https://api-gcp.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        data = fetch_data(url, params)
        if not data or not isinstance(data, list):
            logging.error(f"Failed to get kline data: invalid response - {symbol}, {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Kline data missing required columns: {df.columns.tolist()}")
            return pd.DataFrame()

        df[required_columns] = df[required_columns].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Error processing kline data: {str(e)}")
        return pd.DataFrame()

class BinanceFuturesAnalyzer:
    """Binance Futures Analyzer"""

    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.rate_limiter = RateLimiter(max_requests=5, time_window=1.0)

    def get_usdt_symbols(self) -> List[str]:
        """Get all USDT futures trading symbols"""
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return [symbol['symbol'] for symbol in data['symbols']
                    if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING']
        except Exception as e:
            logging.error(f"Failed to get symbols: {e}")
            return []

    def get_open_interest(self, symbol: str, period: str = "5m") -> Dict:
        """Get open interest data for a specific trading pair"""
        self.rate_limiter.acquire()
        url = f"{self.base_url}/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": 500
        }
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return {"symbol": symbol, "data": data}
        except Exception as e:
            logging.error(f"Failed to get open interest for {symbol}: {e}")
            return None

    def analyze_positions(self) -> pd.DataFrame:
        """Analyze open interest positions for all trading pairs"""
        symbols = self.get_usdt_symbols()
        historical_data = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self.get_open_interest, symbol): symbol
                for symbol in symbols
            }
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result and result['data']:
                        historical_data[symbol] = result['data']
                except Exception as e:
                    logging.error(f"Error processing {symbol} data: {e}")

        analysis_results = []
        for symbol, data in historical_data.items():
            if not data:
                continue
            df = pd.DataFrame(data)
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            except Exception as e:
                logging.error(f"Data processing failed: {e}")
                continue

            current_oi = float(df['sumOpenInterest'].iloc[-1])
            changes = {}
            for period, hours in [('1hours', 1), ('4hours', 4), ('24hours', 24)]:
                try:
                    past_oi = float(df[df['timestamp'] <=
                                       df['timestamp'].max() - pd.Timedelta(hours=hours)]
                                    ['sumOpenInterest'].iloc[-1])
                    change = current_oi - past_oi
                    change_percentage = (change / past_oi) * 100
                    changes[period] = {'change': change, 'change_percentage': change_percentage}
                except (IndexError, ValueError):
                    changes[period] = {'change': 0, 'change_percentage': 0}

            try:
                percentile = (df['sumOpenInterest'].rank(pct=True).iloc[-1]) * 100
            except Exception:
                percentile = 0

            analysis_results.append({
                'symbol': symbol,
                'current_oi': current_oi,
                'percentile': percentile,
                'changes': changes
            })

        df = pd.DataFrame(analysis_results)
        if 'changes' in df.columns:
            df['change_percentage'] = df['changes'].apply(
                lambda x: x.get('24hours', {}).get('change_percentage', 0)
            )
        else:
            df['change_percentage'] = 0
        return df

    async def analyze_market_behavior(self, symbol: str, position_data: dict) -> str:
        """Analyze market behavior and generate an AI report"""
        price_data = {}
        current_price_df = get_binance_klines(symbol, '1m', limit=1)
        if current_price_df.empty or len(current_price_df) < 1:
            error_msg = f"Cannot fetch current price data for {symbol}"
            logging.error(error_msg)
            return error_msg

        current_price = float(current_price_df['close'].iloc[-1])
        periods = [('1h', 1, '1h_change'), ('4h', 4, '4h_change'), ('1d', 24, '24h_change')]

        for period, hours, key in periods:
            df = get_binance_klines(symbol, period, limit=30)
            if df.empty or len(df) < 2:
                error_msg = f"Insufficient kline data for {symbol} {period} period"
                logging.error(error_msg)
                return error_msg

            previous_price = float(df['close'].iloc[-2])
            price_change = ((current_price - previous_price) / previous_price) * 100
            price_data[key] = price_change

        daily_klines = get_binance_klines(symbol, '1d', limit=1)
        if daily_klines.empty or len(daily_klines) < 1:
            error_msg = f"Cannot fetch daily volatility data for {symbol}"
            logging.error(error_msg)
            return error_msg

        high = float(daily_klines['high'].iloc[0])
        low = float(daily_klines['low'].iloc[0])
        volatility = ((high - low) / low) * 100

        required_keys = ['1h_change', '4h_change', '24h_change']
        if not all(key in price_data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in price_data]
            error_msg = f"Missing price data: {missing_keys}"
            logging.error(error_msg)
            return error_msg

        prompt = (
            f"Perform a professional market behavior analysis for {symbol} futures based on:\n\n"
            f"**Market Indicators:**\n"
            f"- Current Open Interest: {position_data['current_oi']:,.0f}\n"
            f"- Open Interest Percentile: {position_data['percentile']:.2f}%\n"
            f"- 24h Volatility: {volatility:.2f}%\n\n"
            f"**Price and OI Comparison:**\n"
            f"| Period | Price Change | OI Change |\n"
            f"|--------|--------------|-----------|\n"
            f"| 1 Hour | {price_data['1h_change']:.2f}% | {position_data['changes']['1hours']['change_percentage']:.2f}% |\n"
            f"| 4 Hours | {price_data['4h_change']:.2f}% | {position_data['changes']['4hours']['change_percentage']:.2f}% |\n"
            f"| 24 Hours | {price_data['24h_change']:.2f}% | {position_data['changes']['24hours']['change_percentage']:.2f}% |\n\n"
            f"Provide an analysis including:\n"
            f"1. **Market behavior patterns**\n"
            f"2. **Bull/Bear strength comparison**\n"
            f"3. **Trading strategy suggestions**\n"
            f"Use markdown, tables, and bold key indicators clearly."
        )

        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"AI report generation failed: {e}"
            logging.error(error_msg)
            return error_msg

    async def generate_position_report(self, df: pd.DataFrame) -> str:
        """Generate an AI report based on open interest analysis"""
        df['change_percentage'] = df.apply(lambda x: x['changes']['24hours']['change_percentage'], axis=1)
        top_increase = df.nlargest(10, 'change_percentage')
        top_decrease = df.nsmallest(10, 'change_percentage')

        prompt = (
            f"Perform a professional market analysis based on Binance futures USDT open interest changes:\n\n"
            f"**Top 10 Increases in Open Interest**\n"
            f"```\n{top_increase[['symbol', 'change_percentage']].to_string()}\n```\n\n"
            f"**Top 10 Decreases in Open Interest**\n"
            f"```\n{top_decrease[['symbol', 'change_percentage']].to_string()}\n```\n\n"
            f"Provide an analysis including:\n"
            f"1. **Market sentiment analysis** based on open interest changes.\n"
            f"2. **Fund flow interpretation**, identifying main categories with inflows/outflows.\n"
            f"3. **Trading strategy recommendations**, highlighting specific trading pairs.\n\n"
            f"Use Markdown format with the structure: [Market Sentiment Analysis] → [Fund Flow Interpretation] → [Trading Strategy Recommendations].\n"
            f"Language concise, professional, actionable signals clearly emphasized.\n"
            f"Use tables for comparisons and highlight key indicators in **bold**."
        )

        try:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"AI report generation failed: {e}")
            return "AI analysis generation failed, please retry later."


async def multi_timeframe_analysis(symbol: str) -> dict:
    """Perform multi-timeframe analysis"""
    results = {}
    trends = {}
    for tf, info in TIMEFRAMES.items():
        df = get_binance_klines(symbol, info['interval'], limit=100)
        if df.empty:
            logging.warning(f"Data fetch failed for timeframe {tf}")
            continue

        df['rsi'] = calculate_rsi(df['close'])
        support, resistance = calculate_support_resistance(df)
        sma20 = df['close'].rolling(window=20).mean()
        sma50 = df['close'].rolling(window=50).mean()
        current_price = float(df['close'].iloc[-1])

        if current_price > sma20.iloc[-1] > sma50.iloc[-1]:
            trend = "Upward"
        elif current_price < sma20.iloc[-1] < sma50.iloc[-1]:
            trend = "Downward"
        else:
            trend = "Sideways"

        volume_sma = df['volume'].rolling(window=20).mean()
        volume_trend = "Increasing" if df['volume'].iloc[-1] > volume_sma.iloc[-1] else "Decreasing"

        trends[tf] = {
            "trend": trend,
            "rsi": df['rsi'].iloc[-1],
            "support": support,
            "resistance": resistance,
            "volume_trend": volume_trend
        }

    if not trends:
        return {"error": "Failed to fetch data for any timeframe"}

    df_current = get_binance_klines(symbol, '1m', limit=1)
    if df_current.empty or 'close' not in df_current.columns:
        logging.error(f"Current price fetch failed: data empty or missing 'close'")
        return {"error": "Failed to fetch current price data"}

    current_price = float(df_current['close'].iloc[0])
    short_term_trend = trends.get('5m', {}).get('trend', '') + "/" + trends.get('15m', {}).get('trend', '')
    medium_term_trend = trends.get('1h', {}).get('trend', '') + "/" + trends.get('4h', {}).get('trend', '')
    rsi_values = [data['rsi'] for data in trends.values()]
    avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 0

    risk = {"level": "Medium", "factors": []}
    if avg_rsi > 70:
        risk["factors"].append("RSI Overbought")
    elif avg_rsi < 30:
        risk["factors"].append("RSI Oversold")

    for tf, data in trends.items():
        if abs(current_price - data['resistance']) / current_price < 0.02:
            risk["factors"].append(f"Close to resistance at {TIMEFRAMES[tf]['name']}")
        if abs(current_price - data['support']) / current_price < 0.02:
            risk["factors"].append(f"Close to support at {TIMEFRAMES[tf]['name']}")

    risk["level"] = "High" if len(risk["factors"]) >= 3 else "Low" if len(risk["factors"]) <= 1 else "Medium"

    # Create trend summary table data
    trend_table_data = []
    for tf in ['5m', '15m', '1h', '4h', '1d']:
        if tf in trends:
            trend_table_data.append({
                "Timeframe": TIMEFRAMES[tf]['name'],
                "Trend": trends[tf]['trend'],
                "RSI": f"{trends[tf]['rsi']:.2f}",
                "Support": f"{trends[tf]['support']:.2f}",
                "Resistance": f"{trends[tf]['resistance']:.2f}",
                "Volume": trends[tf]['volume_trend']
            })

    prompt = (
        f"Generate a multi-timeframe technical analysis report for {symbol}, strictly following this structure:\n\n"
        f"## Market Structure\n"
        f"Describe market **[main characteristic]**:\n"
        f"- **Short-term momentum**: [5m and 15m analysis]\n"
        f"- **Medium-term trend**: [1h and 4h analysis]\n"
        f"- **Long-term trend**: [Daily analysis]\n\n"
        f"Market is in **[structural phase]** with [additional market features]\n\n"
        f"## Technical Analysis\n"
        f"### Key Indicator Validation\n"
        f"1. **Price Structure**:\n"
        f"   - [Support levels]\n"
        f"   - [Resistance levels]\n\n"
        f"2. **Momentum Indicators**:\n"
        f"   - [RSI analysis]\n"
        f"   - [Other momentum indicators]\n\n"
        f"3. **Volume Analysis**:\n"
        f"   - [Price-volume relationship]\n"
        f"   - [Volume trend]\n\n"
        f"## Trading Recommendations\n"
        f"### Short-term strategies (intraday)\n"
        f"**Bullish strategy**:\n"
        f"- Entry: [entry conditions and levels]\n"
        f"- Target: [price targets]\n"
        f"- Stop loss: [stop-loss levels]\n\n"
        f"**Bearish strategy**:\n"
        f"- Entry conditions: [trigger conditions]\n"
        f"- Target: [price targets]\n"
        f"- Stop loss: [stop-loss levels]\n\n"
        f"Base your analysis on:\n"
        f"- Current price: {current_price:.2f}\n"
        f"- Short-term trend (5m/15m): {short_term_trend}\n"
        f"- Medium-term trend (1h/4h): {medium_term_trend}\n"
        f"- RSI indicator: {avg_rsi:.2f}\n"
        f"- Support: {trends.get('1h', {}).get('support', 'N/A'):.2f}\n"
        f"- Resistance: {trends.get('1h', {}).get('resistance', 'N/A'):.2f}\n"
        f"- Risk level: {risk['level']}\n"
        f"- Risk factors: {', '.join(risk['factors']) if risk['factors'] else 'No significant risks'}\n"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a professional crypto analyst. Strictly follow the given format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4000
        )
        ai_analysis = response.choices[0].message.content

        required_sections = ["## Market Structure", "## Technical Analysis", "## Trading Recommendations"]
        for section in required_sections:
            if section not in ai_analysis:
                ai_analysis = ai_analysis.replace(section.replace("##", "#"), section)
    except Exception as e:
        logging.error(f"Multi-timeframe analysis generation failed: {e}")
        ai_analysis = "AI analysis generation failed, please retry later."

    return {
        "current_price": current_price,
        "trends": trends,
        "risk": risk,
        "ai_analysis": ai_analysis
    }

class FundFlowAnalyzer:
    """Fund Flow Analyzer"""

    def __init__(self):
        self.spot_base_url = "https://api.binance.com/api/v3"
        self.futures_base_url = "https://fapi.binance.com/fapi/v1"
        self.stablecoins = {'USDC', 'TUSD', 'BUSD', 'DAI', 'USDP', 'EUR', 'GYEN'}

    def get_all_usdt_symbols(self, is_futures=False):
        """Get all USDT trading pairs excluding stablecoins"""
        base_url = self.futures_base_url if is_futures else self.spot_base_url
        endpoint = "/exchangeInfo"
        response = requests.get(f"{base_url}{endpoint}")
        data = response.json()
        symbols = []
        for item in data['symbols']:
            symbol = item['symbol']
            base_asset = item['baseAsset']
            if (item['status'] == 'TRADING' and
                    item['quoteAsset'] == 'USDT' and
                    base_asset not in self.stablecoins):
                symbols.append(symbol)
        return symbols

    def format_number(self, value):
        """Format numbers to K/M with 2 decimals"""
        if abs(value) >= 1000000:
            return f"{value / 1000000:.2f}M"
        elif abs(value) >= 1000:
            return f"{value / 1000:.2f}K"
        else:
            return f"{value:.2f}"

    def get_klines_parallel(self, symbols, is_futures=False, max_workers=20, include_latest=False):
        """Fetch 4H K-line data in parallel for multiple symbols"""
        results = []
        failed_symbols = []

        def fetch_kline(symbol):
            try:
                base_url = self.futures_base_url if is_futures else self.spot_base_url
                endpoint = "/klines"
                now = datetime.utcnow()
                hours_since_day_start = now.hour + now.minute / 60 + now.second / 3600
                last_4h_offset = int(hours_since_day_start // 4) * 4
                last_4h_start = now.replace(minute=0, second=0, microsecond=0, hour=0) + timedelta(hours=last_4h_offset)
                if last_4h_start > now:
                    last_4h_start -= timedelta(hours=4)
                end_time = int(last_4h_start.timestamp() * 1000)
                start_time = int((last_4h_start - timedelta(hours=4)).timestamp() * 1000)
                params = {
                    'symbol': symbol,
                    'interval': '4h',
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 2
                }
                response = requests.get(f"{base_url}{endpoint}", params=params)
                response.raise_for_status()
                data = response.json()
                if not data or len(data) == 0: 
                    logging.error(f"Insufficient data for {symbol}: only {len(data)} k-lines returned")
                    return None
                k = data[-1]
                result = {
                    'symbol': symbol,
                    'open_time': datetime.fromtimestamp(k[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'close_time': datetime.fromtimestamp(k[6] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'quote_volume': float(k[7]),
                    'trades': int(k[8]),
                    'taker_buy_base_volume': float(k[9]),
                    'taker_buy_quote_volume': float(k[10]),
                    'net_inflow': 2 * float(k[10]) - float(k[7])
                }
                if include_latest:
                    latest_params = {
                        'symbol': symbol,
                        'interval': '1m',
                        'limit': 1
                    }
                    latest_response = requests.get(f"{base_url}{endpoint}", params=latest_params)
                    latest_response.raise_for_status()
                    latest_data = latest_response.json()
                    if latest_data and len(latest_data) > 0:
                        result['latest_price'] = float(latest_data[0][4])
                    else:
                        result['latest_price'] = result['close']
                return result
            except Exception as e:
                logging.error(f"Error fetching {symbol}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_kline, symbol): symbol for symbol in symbols}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_symbols.append(symbol)

        if failed_symbols:
            logging.warning(f"Failed to fetch data for: {failed_symbols}")
        if not results:
            logging.error("No data fetched for any symbol")
            return []
        return results

    async def send_to_chatgpt(self, data):
        """Send data to ChatGPT for generating fund flow analysis"""
        prompt = (
            "You are a professional crypto market analyst. Based on the following Binance spot and futures USDT trading pair data "
            "(last completed 4H interval), please generate a detailed fund flow analysis:\n\n"
            f"{json.dumps(data, indent=2)}\n\n"
            "Instructions:\n"
            "1. **Smart Money Behavior**: Analyze fund inflow/outflow trends across spot and futures markets.\n"
            "2. **Market Stage Assessment**: Determine if the market is trending up, down, or consolidating.\n"
            "3. **Short-Term Outlook (4-8H)**: Predict direction and identify key support/resistance.\n"
            "4. **Trade Strategy Suggestions**: Suggest actions for BTCUSDT, ETHUSDT, etc.\n\n"
            "Use markdown format, highlight important metrics with **bold**, use tables where helpful.\n"
        )

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"ChatGPT API error: {e}")
            return "Failed to get response from ChatGPT."

    def analyze_fund_flow(self):
        """Analyze fund flows"""
        spot_symbols = self.get_all_usdt_symbols(is_futures=False)
        futures_symbols = self.get_all_usdt_symbols(is_futures=True)
        spot_data = self.get_klines_parallel(spot_symbols, is_futures=False, max_workers=20, include_latest=True)
        futures_data = self.get_klines_parallel(futures_symbols, is_futures=True, max_workers=20, include_latest=True)

        spot_df = pd.DataFrame(spot_data)
        futures_df = pd.DataFrame(futures_data)

        if spot_df.empty or 'net_inflow' not in spot_df.columns:
            logging.error("Spot data is empty or missing 'net_inflow'")
            spot_df = pd.DataFrame(columns=['symbol', 'net_inflow', 'quote_volume', 'latest_price'])

        if futures_df.empty or 'net_inflow' not in futures_df.columns:
            logging.error("Futures data is empty or missing 'net_inflow'")
            futures_df = pd.DataFrame(columns=['symbol', 'net_inflow', 'quote_volume', 'latest_price'])

        spot_inflow_top20 = spot_df.sort_values(by='net_inflow', ascending=False).head(20)
        futures_inflow_top20 = futures_df.sort_values(by='net_inflow', ascending=False).head(20)
        spot_outflow_top20 = spot_df.sort_values(by='net_inflow', ascending=True).head(20)
        futures_outflow_top20 = futures_df.sort_values(by='net_inflow', ascending=True).head(20)

        gpt_input_data = {
            "spot_inflow_top20": spot_inflow_top20[['symbol', 'net_inflow', 'quote_volume', 'latest_price']].to_dict('records'),
            "futures_inflow_top20": futures_inflow_top20[['symbol', 'net_inflow', 'quote_volume', 'latest_price']].to_dict('records'),
            "spot_outflow_top20": spot_outflow_top20[['symbol', 'net_inflow', 'quote_volume', 'latest_price']].to_dict('records'),
            "futures_outflow_top20": futures_outflow_top20[['symbol', 'net_inflow', 'quote_volume', 'latest_price']].to_dict('records'),
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }

        analysis = self.send_to_chatgpt(gpt_input_data)
        return {
            "spot_inflow_top20": spot_inflow_top20,
            "futures_inflow_top20": futures_inflow_top20,
            "spot_outflow_top20": spot_outflow_top20,
            "futures_outflow_top20": futures_outflow_top20,
            "analysis": analysis
        }
