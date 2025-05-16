"""
Version 3.0.7 Changes from 3.0.5:
1. Added NeverStopLossInvestor class - A new investor type that holds positions until profitable
2. Enhanced market mechanisms - Improved price calculation and order matching logic
3. Investor behavior improvements - Refined trading strategies for all investor types
4. Bug fixes - Fixed several edge cases in trade execution and price calculations
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

class Investor:
    """Base investor class providing common trading interface
    
    Attributes:
        shares (int): Number of shares currently held
        cash (float): Available cash for trading
    """
    def __init__(self, shares, cash):
        """Initialize investor with shares and cash"""
        self.shares = shares
        self.cash = cash
    def trade(self, price, market):
        """Execute trade at given price in specified market"""
        pass
    def decide_price(self, current_price, market):
        """Determine trading action based on current price and market conditions
        
        Returns:
            tuple: (action, price, shares) where action is 'buy', 'sell' or 'hold'
        """
        return ('hold', 0, 0)

class ValueInvestor(Investor):
    """Value Investor - Trades based on estimated intrinsic value of the stock
    
    Strategy:
    - Sell when market price is above estimated value
    - Buy when market price is below estimated value
    - Trading volume proportional to price deviation percentage
    
    Attributes:
        value_estimate (float): Estimated value of the stock
        k (float): Trading sensitivity coefficient
        estimation_error (float): Standard deviation of estimation error
    """
    def __init__(self, shares, cash, value_estimate, k=1, estimation_error=0.1):
        """Initialize value investor with shares, cash and valuation parameters"""
        super().__init__(shares, cash)
        self.value_estimate = value_estimate  # Estimated intrinsic value
        self.k = k  # Trading sensitivity coefficient
        self.estimation_error = estimation_error  # Valuation error standard deviation
    def decide_price(self, current_price, market):
        diff = (current_price - self.value_estimate) / self.value_estimate
        if diff > 0:
            sell_amount = self.k * diff * self.value_estimate
            sell_shares = min(sell_amount, self.shares)
            if sell_shares > 0:
                sell_price = max(current_price * 0.99, self.value_estimate)
                return ('sell', sell_price, sell_shares)
        elif diff < 0:
            buy_amount = self.k * -diff * self.value_estimate
            buy_shares = min(buy_amount, self.cash / current_price)
            if buy_shares > 0:
                buy_price = min(current_price * 1.01, self.value_estimate)
                return ('buy', buy_price, buy_shares)
        return ('hold', 0, 0)
    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class ChaseInvestor(Investor):
    """Momentum Investor - Follows price trends and momentum
    
    Strategy:
    - Buy when prices are rising (positive momentum)
    - Sell when prices are falling (negative momentum)
    - Uses moving window of N periods to calculate momentum
    
    Attributes:
        N (int): Lookback window size for momentum calculation
    """
    def __init__(self, shares, cash, N=None):
        """Initialize momentum investor with shares, cash and optional window size"""
        super().__init__(shares, cash)
        self.N = N  # Lookback window for momentum calculation
        self._n_initialized = False  # Flag for window size initialization
    def calculate_velocity(self, prices):
        if len(prices) < 2:
            return 0.0
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                change = (prices[i] - prices[i-1]) / prices[i-1]
                price_changes.append(change)
        if not price_changes:
            return 0.0
        return float(sum(price_changes)) / len(price_changes) if price_changes else 0.0
    def decide_price(self, current_price, market):
        if self.N is None and not self._n_initialized:
            self.N = market._rng.choice([3, 5, 10, 15, 20])
            self._n_initialized = True
        if len(market.price_history) >= self.N:
            recent_prices = market.price_history[-self.N:]
            velocity = self.calculate_velocity(recent_prices)
            if velocity > 0:
                buy_ratio = min(abs(velocity) * 5, 1)
                buy_shares = int((self.cash * buy_ratio) / current_price)
                if buy_shares > 0:
                    buy_price = current_price * 1.02
                    return ('buy', buy_price, buy_shares)
            elif velocity < 0:
                sell_ratio = min(abs(velocity) * 5, 1)
                sell_shares = int(self.shares * sell_ratio)
                if sell_shares > 0:
                    sell_price = current_price * 0.98
                    return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)
    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class TrendInvestor(Investor):
    """Trend Following Investor - Trades based on moving average crossover
    
    Strategy:
    - Buy when price crosses above moving average
    - Sell when price crosses below moving average
    - Uses simple moving average of M periods
    
    Attributes:
        M (int): Periods for moving average calculation
        above_ma (bool): Current position relative to moving average
    """
    def __init__(self, shares, cash, M):
        """Initialize trend investor with shares, cash and moving average period"""
        super().__init__(shares, cash)
        self.M = M  # Moving average period
        self.above_ma = None  # Current position relative to MA
    def decide_price(self, current_price, market):
        if len(market.price_history) >= self.M:
            recent_prices = market.price_history[-self.M:]
            if not recent_prices or len(recent_prices) < self.M:
                return ('hold', 0, 0)
            sma = float(sum(recent_prices)) / len(recent_prices)
            current_above_ma = current_price > sma
            if self.above_ma is None:
                self.above_ma = current_above_ma
            elif current_above_ma != self.above_ma:
                self.above_ma = current_above_ma
                if current_above_ma:
                    buy_shares = self.cash // current_price
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        return ('buy', buy_price, buy_shares)
                else:
                    if self.shares > 0:
                        sell_price = current_price * 0.99
                        return ('sell', sell_price, self.shares)
        return ('hold', 0, 0)
    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class RandomInvestor(Investor):
    """Random Trading Investor - Makes random trading decisions
    
    Strategy:
    - Randomly decides to buy/sell with probability p
    - Trades fixed ratio of position/cash
    
    Attributes:
        p (float): Probability of making a trade
        ratio (float): Fraction of position to trade
    """
    def __init__(self, shares, cash, p=0.2, ratio=0.1, seed=None):
        """Initialize random investor with shares, cash and trading parameters"""
        super().__init__(shares, cash)
        self.p = p  # Probability of trading
        self.ratio = ratio  # Fraction of position to trade
        # Initialize random number generator with optional seed
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
    def decide_price(self, current_price, market):
        action = self._rng_investor.choice(['buy', 'sell', 'hold'], p=[self.p, self.p, 1 - 2 * self.p])
        if action == 'buy':
            buy_shares = int(self.cash * self.ratio / current_price)
            if buy_shares > 0 and self.cash >= buy_shares * current_price:
                price_factor = market._rng.uniform(0.95, 1.05)
                buy_price = current_price * price_factor
                return ('buy', buy_price, buy_shares)
        elif action == 'sell':
            sell_shares = int(self.shares * self.ratio)
            if sell_shares > 0:
                price_factor = market._rng.uniform(0.95, 1.05)
                sell_price = current_price * price_factor
                return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)
    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class NeverStopLossInvestor(Investor):
    """Never Stop Loss Investor - Holds positions until profit target reached
    
    Strategy:
    - Rarely buys (low probability)
    - Sells only when position is profitable
    - Never sells at a loss
    
    Attributes:
        buy_price (float): Average purchase price of current position
        buy_probability (float): Probability of buying
        sell_probability (float): Probability of selling when profitable
        profit_target (float): Minimum profit ratio before considering sale
    """
    def __init__(self, shares, cash, buy_probability=0.05, sell_probability=0.5, profit_target=0.1, seed=None):
        """Initialize investor with trading probabilities and profit target"""
        super().__init__(shares, cash)
        self.buy_price = None  # Track average purchase price
        self.buy_probability = buy_probability  # Probability of buying
        self.sell_probability = sell_probability  # Probability of selling when profitable
        self.profit_target = profit_target  # Minimum profit ratio
        # Initialize random number generator with optional seed
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
    def decide_price(self, current_price, market):
        if self.shares > 0 and self.buy_price is not None and current_price > self.buy_price:
            profit_ratio = (current_price - self.buy_price) / self.buy_price
            if 0.05 <= profit_ratio <= 0.50:
                if self._rng_investor.random() < self.sell_probability:
                    sell_price = current_price * 0.99
                    sell_shares = self.shares
                    self.buy_price = None
                    return ('sell', sell_price, sell_shares)
        elif self.cash > 0:
            if self._rng_investor.random() < self.buy_probability:
                buy_shares = int(self.cash / current_price)
                if buy_shares > 0:
                    current_total_cost = (self.shares * self.buy_price if self.buy_price is not None and self.shares > 0 else 0)
                    new_purchase_cost = buy_shares * current_price
                    new_total_shares = self.shares + buy_shares
                    if new_total_shares > 0:
                        self.buy_price = (current_total_cost + new_purchase_cost) / new_total_shares
                    else:
                        pass
                    buy_price = current_price * 1.01
                    return ('buy', buy_price, buy_shares)
        return ('hold', 0, 0)
    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class Market:
    """Stock market simulation environment
    
    Manages price discovery, order matching and trading simulation
    
    Attributes:
        initial_price (float): Starting price of the stock
        price_tick (float): Minimum price increment
        seed (int): Random seed for reproducible simulations
        enable_trade_log (bool): Whether to log trades to file
    """
    def __init__(self, initial_price, price_tick=0.01, seed=None, enable_trade_log=False):
        """Initialize market with starting price and configuration"""
        self.initial_price = initial_price  # Initial stock price
        self.price_tick = price_tick  # Minimum price increment
        self.seed = seed  # Random seed for reproducibility
        self.enable_trade_log = enable_trade_log  # Trade logging flag
        
        # Market state variables
        self.trade_log_file = None  # File for trade logging
        self.price = initial_price  # Current market price
        self.price_history = [initial_price]  # Historical prices
        self.buy_orders = []  # Current buy orders
        self.sell_orders = []  # Current sell orders
        self.executed_volume = 0  # Volume traded in current period
        self.true_value = initial_price  # True underlying value
        self.value_history = [initial_price]  # Historical true values
        
        # Simulation control variables
        self._last_jump_day = 0  # Last day of value jump
        self._next_jump_interval = None  # Days until next value jump
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self._rng_value = np.random.RandomState(self.seed)  # RNG for value changes
        self._rng = np.random.RandomState(self.seed + 9999)  # RNG for other randomness
        self.executed_volume_history = []  # Historical trading volumes
        self.current_day = 0  # Current simulation day
    def init_trade_log(self, filename="trade_log.txt"):
        self.trade_log_file = filename
        if self.enable_trade_log:
            with open(self.trade_log_file, 'w', encoding='utf-8') as f:
                f.write(f"股票市场交易记录 - 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")
                f.write("日期,投资者类型,操作,价格,数量\n")
    def get_investor_type(self, investor):
        if isinstance(investor, ValueInvestor):
            return "Value"
        elif isinstance(investor, ChaseInvestor):
            return "Chase"
        elif isinstance(investor, TrendInvestor):
            return "Trend"
        elif isinstance(investor, RandomInvestor):
            return "Random"
        elif isinstance(investor, NeverStopLossInvestor):
            return "NeverStopLoss"
        return "Unknown"
    def record_trade(self, investor_type, action, price, shares):
        if self.trade_log_file and self.enable_trade_log:
            with open(self.trade_log_file, 'a', encoding='utf-8') as f:
                f.write(f"{self.current_day},{investor_type},{action},{price:.2f},{shares}\n")
    def place_order(self, order_type, price, shares, investor):
        investor_type = self.get_investor_type(investor)
        self.record_trade(investor_type, order_type, price, shares)
        if order_type == 'buy':
            self.buy_orders.append((price, shares, investor))
        elif order_type == 'sell':
            self.sell_orders.append((price, shares, investor))
    def call_auction(self, buy_orders, sell_orders, last_price):
        if not buy_orders or not sell_orders:
            return last_price, 0, set(), set()
        buy_orders_sorted = sorted(enumerate(buy_orders), key=lambda x: x[1][0], reverse=True)
        sell_orders_sorted = sorted(enumerate(sell_orders), key=lambda x: x[1][0])
        possible_prices = sorted(set([order[0] for order in buy_orders + sell_orders]))
        if not possible_prices:
            return last_price, 0, set(), set()
        max_volume = 0
        clearing_price = last_price
        for test_price in possible_prices:
            buy_volume = sum(shares for price, shares, _ in buy_orders if price >= test_price)
            sell_volume = sum(shares for price, shares, _ in sell_orders if price <= test_price)
            executed = min(buy_volume, sell_volume)
            if executed > max_volume:
                max_volume = executed
                clearing_price = test_price
            elif executed == max_volume and abs(test_price - last_price) < abs(clearing_price - last_price):
                clearing_price = test_price
        executed_buy_idx = set()
        executed_sell_idx = set()
        remain_buy = max_volume
        for idx, (price, shares, investor) in buy_orders_sorted:
            if price >= clearing_price and remain_buy > 0:
                exec_shares = min(shares, remain_buy)
                remain_buy -= exec_shares
                executed_buy_idx.add(idx)
        remain_sell = max_volume
        for idx, (price, shares, investor) in sell_orders_sorted:
            if price <= clearing_price and remain_sell > 0:
                exec_shares = min(shares, remain_sell)
                remain_sell -= exec_shares
                executed_sell_idx.add(idx)
        return clearing_price, max_volume, executed_buy_idx, executed_sell_idx
    def execute_trades(self, clearing_price, max_volume, buy_orders, sell_orders, executed_buy_idx, executed_sell_idx):
        if max_volume <= 0 or not buy_orders or not sell_orders or not executed_buy_idx or not executed_sell_idx:
            return
        remain = max_volume
        buy_orders_sorted = sorted(enumerate(buy_orders), key=lambda x: x[1][0], reverse=True)
        if buy_orders_sorted:
            for idx, (price, shares, investor) in buy_orders_sorted:
                if idx in executed_buy_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares
                    investor.shares += exec_shares
                    investor.cash -= exec_shares * clearing_price
                    investor_type = self.get_investor_type(investor)
                    self.record_trade(investor_type, "buy_executed", clearing_price, exec_shares)
                    if hasattr(investor, 'value_estimate'):
                        error_factor = getattr(investor, 'estimation_error', 0.1)
                        investor.value_estimate = self.true_value + self._rng.normal(0, error_factor * max(self.true_value, 0.01))
        remain = max_volume
        sell_orders_sorted = sorted(enumerate(sell_orders), key=lambda x: x[1][0])
        if sell_orders_sorted:
            for idx, (price, shares, investor) in sell_orders_sorted:
                if idx in executed_sell_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares
                    investor.shares -= exec_shares
                    investor.cash += exec_shares * clearing_price
                    investor_type = self.get_investor_type(investor)
                    self.record_trade(investor_type, "sell_executed", clearing_price, exec_shares)
                    if hasattr(investor, 'value_estimate'):
                        error_factor = getattr(investor, 'estimation_error', 0.1)
                        investor.value_estimate = self.true_value + self._rng.normal(0, error_factor * max(self.true_value, 0.01))
    def daily_auction(self):
        self.current_day += 1
        if not self.buy_orders or not self.sell_orders:
            self.price_history.append(self.price)
            self.executed_volume = 0
            self.executed_volume_history.append(0)
            self.value_history.append(self.true_value)
            return
        last_price = self.price_history[-1]
        clearing_price, max_volume, executed_buy_idx, executed_sell_idx = self.call_auction(self.buy_orders, self.sell_orders, last_price)
        self.price = clearing_price
        self.price_history.append(self.price)
        self.executed_volume = max_volume
        self.executed_volume_history.append(self.executed_volume)
        self.execute_trades(clearing_price, max_volume, self.buy_orders, self.sell_orders, executed_buy_idx, executed_sell_idx)
        self.buy_orders = []
        self.sell_orders = []
        current_day = len(self.price_history) - 1
        if self._next_jump_interval is None:
            self._next_jump_interval = self._rng_value.randint(30, 50)
        if current_day - self._last_jump_day >= self._next_jump_interval:
            if self._rng_value.rand() < 0.33:
                change = self._rng_value.uniform(10, 30) * (1 if self._rng_value.rand() < 0.5 else -1)
                self.true_value += change
                if self.true_value < 0:
                    self.true_value = 0
            self._last_jump_day = current_day
            self._next_jump_interval = self._rng_value.randint(30, 50)
        self.value_history.append(self.true_value)

def simulate_stock_market():
    initial_price = 100
    price_tick = 0.01
    days = 2000
    value_line_seed = 2100
    value_investors_params = {
        'num': 150,
        'initial_shares': 200,
        'initial_cash': 15000
    }
    chase_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }
    trend_investors_params = {
        'num': 80,
        'initial_shares': 150,
        'initial_cash': 12000
    }
    random_investors_params = {
        'num': 120,
        'initial_shares': 50,
        'initial_cash': 8000
    }
    never_stop_loss_investors_params = {
        'num': 90,
        'initial_shares': 0,
        'initial_cash': 10000
    }
    market = Market(initial_price, price_tick, value_line_seed)
    import datetime
    trade_log_filename = f"trade_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    market.init_trade_log(trade_log_filename)
    investors = []
    trend_periods = [5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    trend_investors_by_period = {}
    for _ in range(value_investors_params['num']):
        value_estimate = market._rng.normal(market.true_value, 10)
        investors.append(ValueInvestor(
            value_investors_params['initial_shares'],
            value_investors_params['initial_cash'],
            value_estimate
        ))
    for _ in range(chase_investors_params['num']):
        investors.append(ChaseInvestor(
            chase_investors_params['initial_shares'],
            chase_investors_params['initial_cash']
        ))
    trend_investors_num_per_period = trend_investors_params['num'] // len(trend_periods)
    for period in trend_periods:
        trend_investors_by_period[period] = []
        for _ in range(trend_investors_num_per_period):
            investor = TrendInvestor(
                trend_investors_params['initial_shares'],
                trend_investors_params['initial_cash'],
                period
            )
            investors.append(investor)
            trend_investors_by_period[period].append(investor)
    for _ in range(random_investors_params['num']):
        investors.append(RandomInvestor(
            random_investors_params['initial_shares'],
            random_investors_params['initial_cash']
        ))
    for _ in range(never_stop_loss_investors_params['num']):
        investor = NeverStopLossInvestor(
            never_stop_loss_investors_params['initial_shares'],
            never_stop_loss_investors_params['initial_cash'],
            buy_probability=0.2,
            profit_target=0.1
        )
        investors.append(investor)
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']
    prices = [initial_price]
    shares_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': []}
    cash_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': []}
    wealth_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': []}
    trend_assets_by_period = {period: [] for period in trend_periods}
    for _ in range(days):
        for investor in investors:
            investor.trade(market.price, market)
        market.daily_auction()
        type_ranges = [
            ('Value', 0, value_end),
            ('Chase', value_end, chase_end),
            ('Trend', chase_end, trend_end),
            ('Random', trend_end, random_end),
            ('NeverStopLoss', random_end, never_stop_loss_end)
        ]
        for type_name, start, end in type_ranges:
            if start < end:
                shares_list = [inv.shares for inv in investors[start:end]]
                cash_list = [inv.cash for inv in investors[start:end]]
                if shares_list and cash_list:
                    avg_shares = np.mean(shares_list)
                    avg_cash = np.mean(cash_list)
                    avg_wealth = avg_cash + avg_shares * market.price
                else:
                    avg_shares = avg_cash = avg_wealth = 0.0
            else:
                avg_shares = avg_cash = avg_wealth = 0.0
            shares_by_type[type_name].append(avg_shares)
            cash_by_type[type_name].append(avg_cash)
            wealth_by_type[type_name].append(avg_wealth)
        for period in trend_periods:
            investors_list = trend_investors_by_period[period]
            if investors_list:
                assets_list = [inv.cash + inv.shares * market.price for inv in investors_list]
                if assets_list:
                    avg_assets = np.mean(assets_list)
                else:
                    avg_assets = 0.0
            else:
                avg_assets = 0.0
            trend_assets_by_period[period].append(avg_assets)
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
    axs[0].plot(market.price_history, label='Stock Price', color='blue')
    axs[0].plot(market.value_history, label='True Value', linestyle='--', color='green')
    axs[0].set_ylabel('Price', color='blue')
    axs[0].tick_params(axis='y', labelcolor='blue')
    axs[0].legend(loc='upper left')
    ax2 = axs[0].twinx()
    daily_volumes = market.executed_volume_history
    ax2.bar(range(len(daily_volumes)), daily_volumes, alpha=0.3, color='gray')
    ax2.set_ylabel('Volume', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    axs[0].set_title('Stock Price, True Value, and Trading Volume Over Time')
    for type_name in shares_by_type:
        axs[1].plot(shares_by_type[type_name])
    axs[1].set_ylabel('Shares')
    axs[1].set_title('Average Shares Held by Investor Type')
    for type_name in cash_by_type:
        axs[2].plot(cash_by_type[type_name])
    axs[2].set_ylabel('Cash')
    axs[2].set_title('Average Cash Held by Investor Type')
    for type_name in wealth_by_type:
        axs[3].plot(wealth_by_type[type_name])
    axs[3].set_ylabel('Total Wealth')
    axs[3].set_title('Average Total Wealth by Investor Type')
    for period in trend_periods:
        axs[4].plot(trend_assets_by_period[period])
    axs[4].set_ylabel('Total Assets')
    axs[4].set_title('Trend Investors Performance by MA Period')
    axs[4].set_xlabel('Day')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_stock_market()