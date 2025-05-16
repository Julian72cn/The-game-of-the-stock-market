"""
Stock Market Simulation - Version 3.0.4

A simulation program that models the impact of different investor behaviors on stock market prices.
Implements a call auction mechanism for price discovery and trading execution.

Key features:
- Daily call auction recording closing price and trading volume
- Various investor types with different trading strategies
- Random square wave pattern for true stock value changes
- Visualization of simulation results
"""
import numpy as np
import matplotlib.pyplot as plt

class Investor:
    """
    Base Investor class - Defines basic properties and methods for all investors

    Attributes:
        shares: Number of shares held by the investor
        cash: Amount of cash held by the investor
    """
    def __init__(self, shares, cash):
        self.shares = shares  # Number of shares held by the investor
        self.cash = cash      # Amount of cash held by the investor

    def trade(self, price, market):
        """Trading method to be implemented by subclasses"""
        pass

    def decide_price(self, current_price, market):
        """
        Method to decide trading price and quantity, implemented by subclasses

        Returns:
            tuple: (action, price, shares)
                action: 'buy', 'sell' or 'hold'
                price: trading price
                shares: number of shares
        """
        return ('hold', 0, 0)

class ValueInvestor(Investor):
    """
    Value Investor - Trades based on estimated intrinsic value of the stock

    Strategy:
    - Sell when market price is above estimated value
    - Buy when market price is below estimated value
    - Trading volume proportional to price deviation percentage

    Attributes:
        value_estimate: Estimated value of the stock
        k: Trading sensitivity coefficient
        estimation_error: Standard deviation of estimation error
    """
    def __init__(self, shares, cash, value_estimate, k=1, estimation_error=0.1):
        super().__init__(shares, cash)
        self.value_estimate = value_estimate  # Estimated intrinsic value of the stock
        self.k = k  # Trading sensitivity coefficient
        self.estimation_error = estimation_error  # Standard deviation of estimation error

    def decide_price(self, current_price, market):
        """
        Decide trading action based on difference between current price and estimated value
        """
        diff = (current_price - self.value_estimate) / self.value_estimate  # Calculate price deviation percentage
        if diff > 0:  # Current price above estimate, consider selling
            sell_amount = self.k * diff * self.value_estimate  # Sell amount proportional to price deviation percentage
            sell_shares = min(sell_amount, self.shares)
            if sell_shares > 0:
                sell_price = max(current_price * 0.99, self.value_estimate)  # Selling price slightly below market
                return ('sell', sell_price, sell_shares)
        elif diff < 0:  # Current price below estimate, consider buying
            buy_amount = self.k * -diff * self.value_estimate  # Buy amount proportional to price deviation percentage
            buy_shares = min(buy_amount, self.cash / current_price)
            if buy_shares > 0:
                buy_price = min(current_price * 1.01, self.value_estimate)  # Buying price slightly above market
                return ('buy', buy_price, buy_shares)
        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        """
        Execute specific trading operation
        """
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class ChaseInvestor(Investor):
    """
    Chase Investor - Trades based on price change velocity

    Strategy:
    - Buy more when price rises faster, proportional to remaining cash
    - Sell more when price falls faster, proportional to remaining shares
    - Trading volume proportional to price change velocity

    Attributes:
        N: Observation period for calculating price change velocity
    """
    def __init__(self, shares, cash, N=None):
        super().__init__(shares, cash)
        self.N = N  # Observation period for calculating price change velocity
        self._n_initialized = False  # Flag to mark if N has been initialized

    def calculate_velocity(self, prices):
        """
        Calculate price change velocity
        """
        if len(prices) < 2:
            return 0
        # Calculate price change rate
        price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        # Return average change rate
        return sum(price_changes) / len(price_changes)

    def decide_price(self, current_price, market):
        """
        Decide trading action based on price change velocity
        """
        if len(market.price_history) >= self.N:
            recent_prices = market.price_history[-self.N:]  # Get recent price history
            velocity = self.calculate_velocity(recent_prices)  # Calculate price change velocity

            if velocity > 0:  # Price rising trend
                # Buy ratio proportional to velocity, max 80% of cash
                buy_ratio = min(abs(velocity) * 5, 1)  # Convert velocity to buy ratio
                buy_shares = int((self.cash * buy_ratio) / current_price)
                if buy_shares > 0:
                    buy_price = current_price * 1.02  # Accept slightly higher buying price
                    return ('buy', buy_price, buy_shares)
            elif velocity < 0:  # Price falling trend
                # Sell ratio proportional to velocity, max 80% of shares
                sell_ratio = min(abs(velocity) * 5, 1)  # Convert velocity to sell ratio
                sell_shares = int(self.shares * sell_ratio)
                if sell_shares > 0:
                    sell_price = current_price * 0.98  # Accept slightly lower selling price
                    return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        """
        Execute specific trading operation
        """
        if self.N is None and not self._n_initialized:
            self.N = market._rng.choice([3, 5, 10, 15, 20])
            self._n_initialized = True
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class TrendInvestor(Investor):
    """
    Trend Investor - Trades based on moving average

    Strategy:
    - Buy all when price first crosses above moving average
    - Sell all when price first crosses below moving average

    Attributes:
        M: Moving average period
        above_ma: Record if price is above MA
    """
    def __init__(self, shares, cash, M):
        super().__init__(shares, cash)
        self.M = M  # Moving average period for trend detection
        self.above_ma = None  # Records whether price is above moving average

    def decide_price(self, current_price, market):
        """
        Decide trading action based on price's relationship with moving average
        """
        if len(market.price_history) >= self.M:
            # Calculate simple moving average
            sma = sum(market.price_history[-self.M:]) / self.M
            current_above_ma = current_price > sma

            # Detect price crossing MA
            if self.above_ma is None:
                self.above_ma = current_above_ma
            elif current_above_ma != self.above_ma:  # Price crosses MA
                self.above_ma = current_above_ma
                if current_above_ma:  # Crosses above MA, buy all
                    buy_shares = self.cash // current_price  # Calculate maximum shares to buy
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        return ('buy', buy_price, buy_shares)
                else:  # Crosses below MA, sell all
                    if self.shares > 0:
                        sell_price = current_price * 0.99
                        return ('sell', sell_price, self.shares)
        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class RandomInvestor(Investor):
    """
    Random Investor - Simulates irrational trading behavior

    Strategy:
    - Randomly decides to buy or sell
    - Trades at randomly deviated prices
    - Uses fixed ratio for trading

    Attributes:
        p: Probability of buying or selling
        ratio: Fixed ratio for each trade
        seed: Random seed for reproducibility
    """
    def __init__(self, shares, cash, p=0.2, ratio=0.1, seed=None):
        super().__init__(shares, cash)
        self.p = p  # Probability of buying or selling in each period
        self.ratio = ratio  # Fixed ratio of cash/shares for each trade
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))  # Random number generator for this investor

    def decide_price(self, current_price, market):
        """
        Randomly decide trading action and price
        """
        # Randomly choose action: buy, sell or hold
        action = self._rng_investor.choice(['buy', 'sell', 'hold'], p=[self.p, self.p, 1 - 2 * self.p])
        if action == 'buy':
            buy_shares = int(self.cash * self.ratio / current_price)
            if buy_shares > 0 and self.cash >= buy_shares * current_price:
                # Random deviation ±5% from current price
                price_factor = market._rng.uniform(0.95, 1.05)
                buy_price = current_price * price_factor
                return ('buy', buy_price, buy_shares)
        elif action == 'sell':
            sell_shares = int(self.shares * self.ratio)
            if sell_shares > 0:
                # Random deviation ±5% from current price
                price_factor = market._rng.uniform(0.95, 1.05)
                sell_price = current_price * price_factor
                return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        """
        Execute specific trading operation
        """
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class Market:
    """
    Stock Market - Market simulator implementing call auction mechanism

    Strategy:
    - Collect buy and sell orders
    - Determine clearing price through call auction
    - Execute trades based on price priority
    - Stock value changes as a random square wave

    Attributes:
        price: Current market price
        price_tick: Minimum price movement unit
        price_history: Price history
        buy_orders: Buy order queue
        sell_orders: Sell order queue
        executed_volume: Executed trading volume
        true_value: True stock value
        value_history: Value history
        seed: Random seed
    """
    def __init__(self, initial_price, price_tick=0.01, seed=None):
        self.price = initial_price  # Current market price
        self.price_tick = price_tick  # Minimum price movement unit
        self.price_history = [initial_price]  # Historical record of prices
        self.buy_orders = []  # Queue of buy orders
        self.sell_orders = []  # Queue of sell orders
        self.executed_volume = 0  # Current trading volume
        self.true_value = initial_price  # Fundamental value of the stock
        self.value_history = [initial_price]  # Historical record of true values
        self._last_jump_day = 0  # Last day when true value jumped
        self._next_jump_interval = None  # Days until next value jump
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)  # Random seed for reproducibility
        self._rng_value = np.random.RandomState(self.seed)  # RNG for value changes
        self._rng = np.random.RandomState(self.seed + 9999)  # RNG for other market operations
        self.executed_volume_history = []  # Historical record of daily trading volumes

    def place_order(self, order_type, price, shares, investor):
        if order_type == 'buy':
            self.buy_orders.append((price, shares, investor))
        elif order_type == 'sell':
            self.sell_orders.append((price, shares, investor))

    def call_auction(self, buy_orders, sell_orders, last_price):
        if not buy_orders and not sell_orders:
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
        if max_volume <= 0:
            return
        remain = max_volume
        for idx, (price, shares, investor) in sorted(enumerate(buy_orders), key=lambda x: x[1][0], reverse=True):
            if idx in executed_buy_idx and remain > 0:
                exec_shares = min(shares, remain)
                remain -= exec_shares
                investor.shares += exec_shares
                investor.cash -= exec_shares * clearing_price
                if hasattr(investor, 'value_estimate'):
                    investor.value_estimate = self.true_value + self._rng.normal(0, getattr(investor, 'estimation_error', 0.1) * self.true_value)
        remain = max_volume
        for idx, (price, shares, investor) in sorted(enumerate(sell_orders), key=lambda x: x[1][0]):
            if idx in executed_sell_idx and remain > 0:
                exec_shares = min(shares, remain)
                remain -= exec_shares
                investor.shares -= exec_shares
                investor.cash += exec_shares * clearing_price
                if hasattr(investor, 'value_estimate'):
                    investor.value_estimate = self.true_value + self._rng.normal(0, getattr(investor, 'estimation_error', 0.1) * self.true_value)

    def daily_auction(self):
        # Execute one call auction per day to determine price and execute trades
        last_price = self.price_history[-1]
        clearing_price, max_volume, executed_buy_idx, executed_sell_idx = self.call_auction(self.buy_orders, self.sell_orders, last_price)
        self.price = clearing_price
        self.price_history.append(self.price)
        self.executed_volume = max_volume
        self.executed_volume_history.append(self.executed_volume) # Record daily trading volume
        self.execute_trades(clearing_price, max_volume, self.buy_orders, self.sell_orders, executed_buy_idx, executed_sell_idx)
        self.buy_orders = []
        self.sell_orders = []
        # Generate random square wave changes to the true stock value
        current_day = len(self.price_history)
        if self._next_jump_interval is None:
            self._next_jump_interval = self._rng_value.randint(30, 50)  # Random interval between jumps
        if current_day - self._last_jump_day >= self._next_jump_interval:
            if self._rng_value.rand() < 0.33:  # 33% chance of value change
                change = self._rng_value.uniform(10, 30) * (1 if self._rng_value.rand() < 0.5 else -1)  # Random magnitude and direction
                self.true_value += change
                if self.true_value < 0:  # Prevent negative values
                    self.true_value = 0
            self._last_jump_day = current_day
            self._next_jump_interval = self._rng_value.randint(30, 50)  # Set next jump interval
        self.value_history.append(self.true_value)

def simulate_stock_market():
    initial_price = 100
    price_tick = 0.01
    days = 800
    num_per_type = 100
    initial_shares = 100
    initial_cash = 10000
    value_line_seed = 2048

    market = Market(initial_price, price_tick, value_line_seed)

    investors = []
    trend_periods = [5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    trend_investors_by_period = {}

    for _ in range(num_per_type):
        value_estimate = market._rng.normal(market.true_value, 10)
        investors.append(ValueInvestor(initial_shares, initial_cash, value_estimate))
    for _ in range(num_per_type):
        investors.append(ChaseInvestor(initial_shares, initial_cash))
    for period in trend_periods:
        trend_investors_by_period[period] = []
        for _ in range(num_per_type // len(trend_periods)):
            investor = TrendInvestor(initial_shares, initial_cash, period)
            investors.append(investor)
            trend_investors_by_period[period].append(investor)
    for _ in range(num_per_type):
        investors.append(RandomInvestor(initial_shares, initial_cash))

    prices = [initial_price]
    shares_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': []}
    cash_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': []}
    wealth_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': []}
    trend_assets_by_period = {period: [] for period in trend_periods}

    for _ in range(days):
        for investor in investors:
            investor.trade(market.price, market)
        # Execute daily call auction to determine price and execute trades
        market.daily_auction()
        prices.append(market.price)  # Record the new price

        # Calculate and record average holdings, cash and total wealth for each investor type
        for type_name, start, end in [('Value', 0, num_per_type),
                                      ('Chase', num_per_type, 2*num_per_type),
                                      ('Trend', 2*num_per_type, 3*num_per_type),
                                      ('Random', 3*num_per_type, 4*num_per_type)]:
            avg_shares = np.mean([inv.shares for inv in investors[start:end]])  # Average number of shares held
            avg_cash = np.mean([inv.cash for inv in investors[start:end]])      # Average cash holdings
            avg_wealth = avg_cash + avg_shares * market.price                   # Average total wealth (cash + shares value)
            shares_by_type[type_name].append(avg_shares)
            cash_by_type[type_name].append(avg_cash)
            wealth_by_type[type_name].append(avg_wealth)

        # Calculate and record total assets for trend investors with different MA periods
        for period in trend_periods:
            avg_assets = np.mean([inv.cash + inv.shares * market.price for inv in trend_investors_by_period[period]])  # Average total assets
            trend_assets_by_period[period].append(avg_assets)

    # Create visualization to display simulation results
    # Set up 5 subplots for different metrics
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)  # Share x-axis for all subplots

    # Plot stock price and true value (first subplot)
    axs[0].plot(market.price_history, label='Stock Price', color='blue')  # Stock price
    axs[0].plot(market.value_history, label='True Value', linestyle='--', color='green')  # True value
    axs[0].set_ylabel('Price', color='blue')  # Set y-axis label
    axs[0].tick_params(axis='y', labelcolor='blue')  # Set tick color
    axs[0].legend(loc='upper left')  # Add legend

    # Add trading volume (second y-axis) to first subplot
    ax2 = axs[0].twinx()  # Create twin y-axis sharing x-axis
    # Note: Volume recording needs adjustment, temporarily using length of price_history for x-axis
    daily_volumes = [market.executed_volume_history[i] if i < len(market.executed_volume_history) else 0 for i in range(days)]
    ax2.bar(range(days), daily_volumes, alpha=0.3, color='gray', label='Trading Volume')  # Volume bar chart
    ax2.set_ylabel('Volume', color='gray')  # Set y-axis label
    ax2.tick_params(axis='y', labelcolor='gray')  # Set tick color
    ax2.legend(loc='upper right')  # Add legend
    axs[0].set_title('Stock Price, True Value, and Trading Volume Over Time')  # Set title

    # Plot average shares by investor type (second subplot)
    for type_name in shares_by_type:
        axs[1].plot(shares_by_type[type_name], label=f'{type_name} Investors')  # Plot shares curve
    axs[1].set_ylabel('Shares')  # Set y-axis label
    axs[1].set_title('Average Shares Held by Investor Type')  # Set title
    axs[1].legend()  # Add legend

    # Plot average cash by investor type (third subplot)
    for type_name in cash_by_type:
        axs[2].plot(cash_by_type[type_name], label=f'{type_name} Investors')  # Plot cash curve
    axs[2].set_ylabel('Cash')  # Set y-axis label
    axs[2].set_title('Average Cash Held by Investor Type')  # Set title
    axs[2].legend()  # Add legend

    # Plot average total wealth by investor type (fourth subplot)
    for type_name in wealth_by_type:
        axs[3].plot(wealth_by_type[type_name], label=f'{type_name} Investors')  # Plot wealth curve
    axs[3].set_ylabel('Total Wealth')  # Set y-axis label
    axs[3].set_title('Average Total Wealth by Investor Type')  # Set title
    axs[3].legend()  # Add legend

    # Plot trend investors' performance by period (fifth subplot)
    for period in trend_periods:
        axs[4].plot(trend_assets_by_period[period], label=f'MA{period}')  # Plot assets curve
    axs[4].set_ylabel('Total Assets')  # Set y-axis label
    axs[4].set_title('Trend Investors Performance by MA Period')  # Set title
    axs[4].legend()  # Add legend

    # Set common x-label
    axs[4].set_xlabel('Day')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Run simulation
if __name__ == "__main__":
    simulate_stock_market()
