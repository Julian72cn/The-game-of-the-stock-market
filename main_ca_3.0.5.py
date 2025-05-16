"""  
Stock Market Simulation - Version 3.0.5

A simulation program that models the impact of different investor behaviors on stock market prices.
Implements a call auction mechanism for price discovery and trading execution.

Changes from Version 3.0.4:
1. Enhanced ChaseInvestor's velocity calculation:
   - Added robustness against zero price values
   - Improved price change calculation with explicit list building
   - Added explicit float conversion for more precise calculations

2. Optimized TrendInvestor's decision making:
   - Added validation for recent price data availability
   - Improved moving average calculation with explicit float conversion

3. Refined Market class functionality:
   - Added executed_volume_history tracking
   - Enhanced random number generation with separate seeds
   - Improved order matching algorithm efficiency

Key features:
- Daily call auction recording closing price and trading volume
- Various investor types with different trading strategies
- Random value jumps for true stock value changes
- Comprehensive visualization of simulation results
"""
import numpy as np
import matplotlib.pyplot as plt

class Investor:
    """Base Investor class - Defines basic properties and methods for all investors
    
    Attributes:
        shares (int): Number of shares held by the investor
        cash (float): Amount of cash held by the investor
    """
    def __init__(self, shares, cash):
        self.shares = shares
        self.cash = cash

    def trade(self, price, market):
        """Trading method to be implemented by subclasses"""
        pass

    def decide_price(self, current_price, market):
        """Method to decide trading price and quantity, implemented by subclasses
        
        Returns:
            tuple: (action, price, shares)
                action: 'buy', 'sell' or 'hold'
                price: trading price
                shares: number of shares
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
        super().__init__(shares, cash)
        self.value_estimate = value_estimate
        self.k = k
        self.estimation_error = estimation_error

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
    """Chase Investor - Trades based on price change velocity
    
    Strategy:
    - Buy when price rises, with volume proportional to velocity
    - Sell when price falls, with volume proportional to velocity
    - Random observation period selection for price trend analysis
    
    Attributes:
        N (int): Observation period for calculating price change velocity
        _n_initialized (bool): Flag to track if N has been initialized
    """
    def __init__(self, shares, cash, N=None):
        super().__init__(shares, cash)
        self.N = N
        self._n_initialized = False

    def calculate_velocity(self, prices):
        """Calculate average price change velocity with safeguards against zero prices
        
        Args:
            prices (list): List of historical prices
            
        Returns:
            float: Average price change velocity
        """
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
            self.N = market._rng.choice([ 3, 5, 10, 15, 20])
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
    """Trend Investor - Trades based on moving average crossovers
    
    Strategy:
    - Buy all when price crosses above moving average
    - Sell all when price crosses below moving average
    - Uses simple moving average for trend detection
    
    Attributes:
        M (int): Moving average period
        above_ma (bool): Tracks if price is above moving average
    """
    def __init__(self, shares, cash, M):
        super().__init__(shares, cash)
        self.M = M
        self.above_ma = None

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
    """Random Investor - Trades based on random decisions
    
    Strategy:
    - Random buy/sell decisions with specified probability
    - Trading volume determined by ratio parameter
    - Random price adjustments within specified range
    
    Attributes:
        p (float): Probability of trading
        ratio (float): Portion of holdings to trade
        _rng_investor (RandomState): Random number generator
    """
    def __init__(self, shares, cash, p=0.2, ratio=0.1, seed=None):
        super().__init__(shares, cash)
        self.p = p
        self.ratio = ratio
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

class Market:
    """Market class - Implements call auction mechanism and price discovery
    
    Attributes:
        price (float): Current market price
        price_tick (float): Minimum price movement
        price_history (list): Historical price records
        buy_orders (list): Current buy orders
        sell_orders (list): Current sell orders
        executed_volume (int): Volume executed in current auction
        executed_volume_history (list): Historical trading volumes
        true_value (float): Underlying true value of the stock
        value_history (list): Historical true values
        seed (int): Random seed for reproducibility
    """
    def __init__(self, initial_price, price_tick=0.01, seed=None):
        self.price = initial_price
        self.price_tick = price_tick
        self.price_history = [initial_price]
        self.buy_orders = []
        self.sell_orders = []
        self.executed_volume = 0
        self.true_value = initial_price
        self.value_history = [initial_price]
        self._last_jump_day = 0
        self._next_jump_interval = None
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self._rng_value = np.random.RandomState(self.seed)
        self._rng = np.random.RandomState(self.seed + 9999)
        self.executed_volume_history = []

    def place_order(self, order_type, price, shares, investor):
        """Place a new order in the market
        
        Args:
            order_type (str): 'buy' or 'sell'
            price (float): Order price
            shares (int): Number of shares
            investor (Investor): Order placer
        """
        if order_type == 'buy':
            self.buy_orders.append((price, shares, investor))
        elif order_type == 'sell':
            self.sell_orders.append((price, shares, investor))

    def call_auction(self, buy_orders, sell_orders, last_price):
        """Execute call auction to determine clearing price and executed orders
        
        Args:
            buy_orders (list): List of buy orders
            sell_orders (list): List of sell orders
            last_price (float): Previous clearing price
            
        Returns:
            tuple: (clearing_price, volume, executed_buy_orders, executed_sell_orders)
        """
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
        """Execute matched trades and update investor positions
        
        Args:
            clearing_price (float): Price at which trades are executed
            max_volume (int): Total volume to be executed
            buy_orders (list): List of buy orders
            sell_orders (list): List of sell orders
            executed_buy_idx (set): Indices of executed buy orders
            executed_sell_idx (set): Indices of executed sell orders
        """
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
                    if hasattr(investor, 'value_estimate'):
                        error_factor = getattr(investor, 'estimation_error', 0.1)
                        investor.value_estimate = self.true_value + self._rng.normal(0, error_factor * max(self.true_value, 0.01))

    def daily_auction(self):
        """Execute daily call auction and update market state
        
        This method:
        1. Determines clearing price through call auction
        2. Executes trades at clearing price
        3. Updates true value with random jumps
        4. Records price and volume history
        """
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
    """Run stock market simulation with multiple investor types
    
    This function:
    1. Initializes market and different types of investors
    2. Runs daily trading simulation
    3. Tracks and visualizes various metrics:
       - Price and volume
       - Investor holdings and performance
       - Strategy comparison
    """
    initial_price = 100
    price_tick = 0.01
    days = 500
    value_line_seed = 2100

    value_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }
    
    chase_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }
    
    trend_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }
    
    random_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    market = Market(initial_price, price_tick, value_line_seed)

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

    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']

    prices = [initial_price]
    shares_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': []}
    cash_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': []}
    wealth_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': []}
    trend_assets_by_period = {period: [] for period in trend_periods}

    for _ in range(days):
        for investor in investors:
            investor.trade(market.price, market)
        market.daily_auction()

        type_ranges = [
            ('Value', 0, value_end),
            ('Chase', value_end, chase_end),
            ('Trend', chase_end, trend_end),
            ('Random', trend_end, random_end)
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
    ax2.bar(range(len(daily_volumes)), daily_volumes, alpha=0.3, color='gray', label='Trading Volume')
    ax2.set_ylabel('Volume', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')
    axs[0].set_title('Stock Price, True Value, and Trading Volume Over Time')
    
    for type_name in shares_by_type:
        axs[1].plot(shares_by_type[type_name], label=f'{type_name} Investors')
    axs[1].set_ylabel('Shares')
    axs[1].set_title('Average Shares Held by Investor Type')
    axs[1].legend()
    
    for type_name in cash_by_type:
        axs[2].plot(cash_by_type[type_name], label=f'{type_name} Investors')
    axs[2].set_ylabel('Cash')
    axs[2].set_title('Average Cash Held by Investor Type')
    axs[2].legend()
    
    for type_name in wealth_by_type:
        axs[3].plot(wealth_by_type[type_name], label=f'{type_name} Investors')
    axs[3].set_ylabel('Total Wealth')
    axs[3].set_title('Average Total Wealth by Investor Type')
    axs[3].legend()
    
    for period in trend_periods:
        axs[4].plot(trend_assets_by_period[period], label=f'MA{period}')
    axs[4].set_ylabel('Total Assets')
    axs[4].set_title('Trend Investors Performance by MA Period')
    axs[4].legend()
    
    axs[4].set_xlabel('Day')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_stock_market()
