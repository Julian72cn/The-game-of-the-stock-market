# Version 3.0.8 Upgrade Notes
# 
# Major Improvements:
# 1. Enhanced Documentation:
#    - Added comprehensive docstrings for all classes and methods
#    - Improved code comments with detailed explanations
#    - Added strategy descriptions for each investor type
#
# 2. Market Mechanism Enhancements:
#    - Improved call auction algorithm for better price discovery
#    - Enhanced order matching logic with sorted order execution
#    - Added volume tracking and history
#    - Implemented more realistic true value evolution
#
# 3. Investor Classes Optimization:
#    - Refined trading strategies with better parameter handling
#    - Added error checking and edge case handling
#    - Improved random number generation with separate seeds
#    - Enhanced position tracking and risk management
#
# 4. Simulation Framework Improvements:
#    - Added detailed performance tracking by investor type
#    - Improved visualization with multiple metrics
#    - Better initialization of simulation parameters
#    - More realistic market dynamics modeling
# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Base class for all investors
class Investor:
    def __init__(self, shares, cash):
        self.shares = shares  # Number of shares held by the investor
        self.cash = cash      # Amount of cash held by the investor

    def trade(self, price, market):
        # Abstract method for executing trades, to be implemented by subclasses
        pass

    def decide_price(self, current_price, market):
        # Abstract method for deciding trade action, to be implemented by subclasses
        # Returns a tuple: (action, price, shares)
        return ('hold', 0, 0)

class ValueInvestor(Investor):
    """
    Value Investor - Trades based on estimated intrinsic value of the stock
    
    Strategy:
    - Sell when market price is above estimated value
    - Buy when market price is below estimated value
    - Trading volume proportional to price deviation percentage
    
    Attributes:
        value_estimate: Estimated intrinsic value of the stock
        k: Trading sensitivity coefficient (higher k = more aggressive trades)
        estimation_error: Standard deviation of estimation error
    """
    def __init__(self, shares, cash, value_estimate, k=1, estimation_error=0.1):
        super().__init__(shares, cash)
        self.value_estimate = value_estimate  # Estimated intrinsic value of the stock
        self.k = k  # Sensitivity coefficient for trading volume
        self.estimation_error = estimation_error  # Standard deviation for estimation error

    def decide_price(self, current_price, market):
        """
        Decide trading action based on difference between current price and estimated value
        
        Args:
            current_price: Current market price of the stock
            market: Market reference (unused in this strategy)
            
        Returns:
            tuple: (action, price, shares)
                action: 'buy', 'sell' or 'hold'
                price: proposed trading price
                shares: number of shares to trade
        """
        # Calculate percentage deviation from estimated value
        diff = (current_price - self.value_estimate) / self.value_estimate
        
        if diff > 0:  # Overpriced - consider selling
            sell_amount = self.k * diff * self.value_estimate  # Sell amount proportional to price deviation
            sell_shares = min(sell_amount, self.shares)  # Can't sell more than owned
            if sell_shares > 0:
                # Sell at slightly below market price or estimated value (whichever is higher)
                sell_price = max(current_price * 0.99, self.value_estimate)
                return ('sell', sell_price, sell_shares)
                
        elif diff < 0:  # Underpriced - consider buying
            buy_amount = self.k * -diff * self.value_estimate  # Buy amount proportional to price deviation
            buy_shares = min(buy_amount, self.cash / current_price)  # Can't buy more than cash allows
            if buy_shares > 0:
                # Buy at slightly above market price or estimated value (whichever is lower)
                buy_price = min(current_price * 1.01, self.value_estimate)
                return ('buy', buy_price, buy_shares)
                
        return ('hold', 0, 0)  # No significant price deviation - hold position

    def trade(self, price, market):
        """
        Execute trading operation by placing orders in the market
        
        Args:
            price: Current market price (unused, uses decide_price instead)
            market: Market object to place orders with
        """
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class ChaseInvestor(Investor):
    """
    Chase Investor - Uses momentum/velocity based trading strategy
    
    Strategy:
    - Calculates price velocity over N periods
    - Buys when price is rising (positive velocity)
    - Sells when price is falling (negative velocity)
    - Trade size proportional to velocity magnitude
    
    Attributes:
        N: Number of periods to observe for velocity calculation
        _n_initialized: Flag to check if N is initialized
    """
    def __init__(self, shares, cash, N=None):
        super().__init__(shares, cash)
        self.N = N  # Number of periods to observe for velocity calculation
        self._n_initialized = False  # Flag to check if N is initialized

    def calculate_velocity(self, prices):
        """
        Calculate the average rate of price change (velocity) over the given price history
        
        Args:
            prices: List of historical prices to calculate velocity from
            
        Returns:
            float: Average price change rate (velocity). Positive means upward trend,
                  negative means downward trend. Returns 0.0 if not enough price data
                  or no valid changes can be calculated.
        """
        if len(prices) < 2:  # Need at least 2 prices to calculate change
            return 0.0
        price_changes = []  # Store percentage changes between consecutive prices
        for i in range(1, len(prices)):
            if prices[i-1] > 0:  # Avoid division by zero
                change = (prices[i] - prices[i-1]) / prices[i-1]  # Calculate percentage change
                price_changes.append(change)
        # Return average of price changes if any exist, otherwise 0.0
        return float(sum(price_changes)) / len(price_changes) if price_changes else 0.0

    def decide_price(self, current_price, market):
        """
        Decide trading action based on recent price velocity
        
        Args:
            current_price: Current market price of the stock
            market: Market object containing price history
            
        Returns:
            tuple: (action, price, shares)
                action: 'buy' if velocity > 0, 'sell' if velocity < 0, 'hold' otherwise
                price: Proposed trading price with premium/discount based on action
                shares: Number of shares to trade, proportional to velocity magnitude
        """
        # Initialize N if not already set
        if self.N is None and not self._n_initialized:
            self.N = market._rng.choice([3, 5, 10, 15, 20])  # Choose a random window size
            self._n_initialized = True
            
        if len(market.price_history) >= self.N:
            recent_prices = market.price_history[-self.N:]  # Get most recent N prices
            velocity = self.calculate_velocity(recent_prices)
            
            if velocity > 0:  # Upward trend detected
                buy_ratio = min(abs(velocity) * 5, 1)  # Buy amount proportional to velocity
                buy_shares = int((self.cash * buy_ratio) / current_price)
                if buy_shares > 0:
                    buy_price = current_price * 1.02  # Willing to pay 2% premium in uptrend
                    return ('buy', buy_price, buy_shares)
            elif velocity < 0:  # Downward trend detected
                sell_ratio = min(abs(velocity) * 5, 1)  # Sell amount proportional to velocity
                sell_shares = int(self.shares * sell_ratio)
                if sell_shares > 0:
                    sell_price = current_price * 0.98  # Accept 2% discount in downtrend
                    return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)  # No trade if insufficient price history or no trend

    def trade(self, price, market):
        """
        Execute trading operation by placing orders in the market
        
        Args:
            price: Current market price (unused, uses decide_price instead)
            market: Market object to place orders with
        """
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class TrendInvestor(Investor):
    """
    Trend Investor - Uses moving average crossover strategy
    
    Strategy:
    - Calculates simple moving average (SMA) over M periods
    - Buys when price crosses above SMA (bullish signal)
    - Sells when price crosses below SMA (bearish signal)
    - Uses all available cash when buying, sells all shares when selling
    
    Attributes:
        M: Moving average window size
        above_ma: Boolean tracking if price is currently above moving average
    """
    def __init__(self, shares, cash, M):
        super().__init__(shares, cash)
        self.M = M  # Moving average window size
        self.above_ma = None  # Track if current price is above MA

    def decide_price(self, current_price, market):
        """
        Decide trading action based on price crossing the moving average
        
        Args:
            current_price: Current market price of the stock
            market: Market object containing price history
            
        Returns:
            tuple: (action, price, shares)
                action: 'buy' when crossing above MA, 'sell' when crossing below, 'hold' otherwise
                price: Proposed trading price with slight premium/discount from market price
                shares: Number of shares to trade (all available cash/shares)
        """
        if len(market.price_history) >= self.M:
            recent_prices = market.price_history[-self.M:]
            if not recent_prices or len(recent_prices) < self.M:  # Ensure enough price data
                return ('hold', 0, 0)
            sma = float(sum(recent_prices)) / len(recent_prices)  # Calculate simple moving average
            current_above_ma = current_price > sma  # Check if price is above MA
            
            # Detect MA crossover
            if self.above_ma is None:
                self.above_ma = current_above_ma  # Initialize state
            elif current_above_ma != self.above_ma:  # MA crossover detected
                self.above_ma = current_above_ma
                if current_above_ma:  # Price crosses above MA - bullish signal
                    buy_shares = self.cash // current_price  # Use all available cash
                    if buy_shares > 0:
                        buy_price = current_price * 1.01  # Willing to pay 1% premium
                        return ('buy', buy_price, buy_shares)
                else:  # Price crosses below MA - bearish signal
                    if self.shares > 0:
                        sell_price = current_price * 0.99  # Accept 1% discount
                        return ('sell', sell_price, self.shares)
        return ('hold', 0, 0)

    def trade(self, price, market):
        """
        Execute trading operation by placing orders in the market
        
        Args:
            price: Current market price (unused, uses decide_price instead)
            market: Market object to place orders with
        """
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class RandomInvestor(Investor):
    """
    Random Investor - Simulates irrational trading behavior
    
    Strategy:
    - Makes random buy/sell decisions with fixed probability
    - Uses fixed proportion of cash/shares for each trade
    - Trades at randomly deviated prices from current price
    
    Attributes:
        p: Probability of initiating a buy or sell action
        ratio: Fixed proportion of cash/shares to use in each trade
        _rng_investor: Random number generator for consistent randomization
    """
    def __init__(self, shares, cash, p=0.2, ratio=0.1, seed=None):
        super().__init__(shares, cash)
        self.p = p  # Trading probability (p for buy, p for sell, 1-2p for hold)
        self.ratio = ratio  # Fixed proportion of cash/shares to trade with
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        """
        Randomly decide trading action and price
        
        Args:
            current_price: Current market price of the stock
            market: Market object (used for random price deviation)
            
        Returns:
            tuple: (action, price, shares)
                action: 'buy', 'sell', or 'hold' chosen randomly
                price: Current price with random ±5% deviation
                shares: Fixed ratio of cash/shares to trade
        """
        action = self._rng_investor.choice(['buy', 'sell', 'hold'], p=[self.p, self.p, 1 - 2 * self.p])
        if action == 'buy':
            buy_shares = int(self.cash * self.ratio / current_price)
            if buy_shares > 0 and self.cash >= buy_shares * current_price:
                price_factor = market._rng.uniform(0.95, 1.05)  # Random price factor ±5%
                buy_price = current_price * price_factor
                return ('buy', buy_price, buy_shares)
        elif action == 'sell':
            sell_shares = int(self.shares * self.ratio)
            if sell_shares > 0:
                price_factor = market._rng.uniform(0.95, 1.05)  # Random price factor ±5%
                sell_price = current_price * price_factor
                return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)

    def trade(self, price, market):
        """
        Execute trading operation by placing orders in the market
        
        Args:
            price: Current market price (unused, uses decide_price instead)
            market: Market object to place orders with
        """
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class NeverStopLossInvestor(Investor):
    """
    Never Stop Loss Investor - Holds positions until reaching profit target
    
    Strategy:
    - Randomly decides to buy in with all available cash
    - Never sells at a loss, holds position indefinitely if in loss
    - Only sells when reaching a predefined profit target
    - Resets after selling and starts looking for new opportunities
    
    Attributes:
        buy_price: Entry price for current position (None if not holding)
        buy_probability: Probability of initiating a new position
        profit_target: Target profit percentage before selling
        _rng_investor: Random number generator for decision making
    """
    def __init__(self, shares, cash, buy_probability=0.05, profit_target=0.1, seed=None):
        super().__init__(shares, cash)
        self.buy_price = None  # Record the price at which shares were bought
        self.buy_probability = buy_probability  # Probability to buy when not holding shares
        self.profit_target = profit_target  # Target profit ratio for selling
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        """
        Decide trading action based on current position and profit target
        
        Args:
            current_price: Current market price of the stock
            market: Market object (unused in this strategy)
            
        Returns:
            tuple: (action, price, shares)
                action: 'buy' if random trigger and no position, 'sell' if profit target reached
                price: Entry price +1% for buys, current price -1% for sells
                shares: All available cash for buys, all shares for sells
        """
        if self.shares == 0 and self.buy_price is None:  # No current position
            if self._rng_investor.random() < self.buy_probability:  # Random buy trigger
                buy_shares = int(self.cash / current_price)
                if buy_shares > 0:
                    self.buy_price = current_price  # Record entry price
                    buy_price = current_price * 1.01  # Willing to pay slight premium
                    return ('buy', buy_price, buy_shares)
        
        elif self.shares > 0 and self.buy_price is not None:  # Have position
            profit_ratio = (current_price - self.buy_price) / self.buy_price
            
            if profit_ratio >= self.profit_target:  # Reached profit target
                sell_price = current_price * 0.99  # Accept slight discount
                sell_shares = self.shares  # Sell entire position
                self.buy_price = None  # Reset entry price
                return ('sell', sell_price, sell_shares)
                
        return ('hold', 0, 0)  # Either no opportunity or holding for profit

    def trade(self, price, market):
        """
        Execute trading operation by placing orders in the market
        
        Args:
            price: Current market price (unused, uses decide_price instead)
            market: Market object to place orders with
        """
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class BottomFishingInvestor(Investor):
    """
    Bottom Fishing Investor - Implements a strategy to buy during price drops
    
    Strategy:
    - Monitors price drops from recent peak
    - Initiates buying when price drops beyond trigger threshold
    - Increases position size as price drops further (step-wise accumulation)
    - Sells entire position when profit target is reached
    
    Attributes:
        avg_cost: Average cost basis of current position
        profit_target: Target profit percentage for selling (randomly set between 10-50%)
        trigger_drop: Initial price drop percentage to start buying (5-15%)
        step_drop: Additional drop percentage for increasing position size (5-15%)
        last_buy_price: Price at which last purchase was made
    """
    def __init__(self, shares, cash, profit_target=None, seed=None):
        """
        Initialize a Bottom Fishing Investor
        
        Args:
            shares: Initial number of shares
            cash: Initial cash balance
            profit_target: Target profit ratio for selling, randomly set if None
            seed: Random seed for reproducibility
        """
        super().__init__(shares, cash)
        self.avg_cost = None
        self.profit_target = profit_target if profit_target is not None else np.random.uniform(0.1, 0.5)
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.trigger_drop = self._rng_investor.uniform(0.05, 0.15)  # Initial drop to trigger buying
        self.step_drop = self._rng_investor.uniform(0.05, 0.15)    # Additional drop for position sizing
        self.last_buy_price = None

    def decide_price(self, current_price, market):
        """
        Decide trading action based on price drops and profit targets
        
        Strategy details:
        1. Check for profit target:
           - If holding position and profit target reached, sell entire position
        2. Monitor price drops:
           - Look at last 100 days of price history
           - Calculate drop from peak price
           - Buy more aggressively as price drops further
        3. Position sizing:
           - Buy ratio increases with number of price drop steps
           - Maximum position size limited to 80% of available cash
           
        Args:
            current_price: Current market price of the stock
            market: Market object containing price history
            
        Returns:
            tuple: (action, price, shares)
                action: 'sell' if profit target reached, 'buy' if significant drop, 'hold' otherwise
                price: Current price ±1% depending on action
                shares: Based on drop magnitude for buys, all shares for sells
        """
        # First check if profit target is reached
        if self.shares > 0 and self.avg_cost is not None:
            profit_ratio = (current_price - self.avg_cost) / self.avg_cost
            if profit_ratio >= self.profit_target:
                sell_price = current_price * 0.99  # Accept 1% discount
                return ('sell', sell_price, self.shares)
                
        # Look for buying opportunities based on price drops
        if len(market.price_history) >= 100:  # Need sufficient history
            peak_price = max(market.price_history[-100:])  # Find recent peak
            drop_from_peak = (peak_price - current_price) / peak_price
            
            if drop_from_peak >= self.trigger_drop:  # Price dropped enough to consider buying
                # Calculate how many additional drop steps have occurred
                drop_steps = int((drop_from_peak - self.trigger_drop) / self.step_drop)
                if drop_steps > 0:
                    # Increase buy ratio with more drop steps (10% base + 10% per step, max 80%)
                    buy_ratio = min(0.8, 0.1 + drop_steps * 0.1)
                    buy_shares = int((self.cash * buy_ratio) / current_price)
                    if buy_shares > 0:
                        buy_price = current_price * 1.01  # Pay up to 1% premium
                        return ('buy', buy_price, buy_shares)
                        
        return ('hold', 0, 0)  # No action needed

    def trade(self, price, market):
        """
        Execute trading operation and update position tracking
        
        Strategy:
        - For buy orders: Calculates new average cost basis after purchase
        - For sell orders: Resets position tracking after full sale
        
        Args:
            price: Current market price (unused, uses decide_price instead)
            market: Market object to place orders with
        """
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            # Calculate new average cost basis including this purchase
            total_cost = (self.avg_cost * self.shares if self.avg_cost is not None else 0) + order_price * shares
            total_shares = self.shares + shares
            self.avg_cost = total_cost / total_shares if total_shares > 0 else None
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)
            self.avg_cost = None  # Reset cost basis after selling

class Market:
    """
    Market - Simulates a stock market with price discovery through daily auctions
    
    Core functionality:
    - Maintains order books for buy and sell orders
    - Implements price discovery through daily call auctions
    - Tracks price history and trading volume
    - Models underlying true value with random jumps
    
    Attributes:
        price: Current market price of the stock
        price_tick: Minimum price movement increment
        price_history: List of historical prices
        buy_orders: List of pending buy orders (price, shares, investor)
        sell_orders: List of pending sell orders
        executed_volume: Number of shares traded in last auction
        true_value: Underlying fundamental value of the stock
        value_history: List of historical true values
        executed_volume_history: List of historical trading volumes
    """
    def __init__(self, initial_price, price_tick=0.01, seed=None):
        """
        Initialize market with starting price and parameters
        
        Args:
            initial_price: Starting price of the stock
            price_tick: Minimum price increment (default: 0.01)
            seed: Random seed for reproducibility
        """
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
        """
        Add a new order to the appropriate order book
        
        Args:
            order_type: 'buy' or 'sell'
            price: Limit price for the order
            shares: Number of shares to trade
            investor: Reference to the investor placing the order
        """
        if order_type == 'buy':
            self.buy_orders.append((price, shares, investor))
        elif order_type == 'sell':
            self.sell_orders.append((price, shares, investor))

    def call_auction(self, buy_orders, sell_orders, last_price):
        """
        Find clearing price and executed volume through price discovery
        
        Steps:
        1. Sort buy orders by price (highest first) and sell orders by price (lowest first)
        2. For each possible price, calculate potential trading volume
        3. Select price that maximizes trading volume
        4. If multiple prices give same volume, choose closest to last price
        5. Determine which orders will execute
        
        Args:
            buy_orders: List of buy orders (price, shares, investor)
            sell_orders: List of sell orders
            last_price: Previous market price
            
        Returns:
            tuple: (clearing_price, max_volume, executed_buy_idx, executed_sell_idx)
                clearing_price: Price at which trades will execute
                max_volume: Number of shares that will trade
                executed_buy_idx: Indices of executed buy orders
                executed_sell_idx: Indices of executed sell orders
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
            
            # Update clearing price if this price gives more volume or same volume closer to last price
            if executed > max_volume:
                max_volume = executed
                clearing_price = test_price
            elif executed == max_volume and abs(test_price - last_price) < abs(clearing_price - last_price):
                clearing_price = test_price
                
        # Determine which orders will execute
        executed_buy_idx = set()
        executed_sell_idx = set()
        
        # Fill buy orders from highest price down
        remain_buy = max_volume
        for idx, (price, shares, investor) in buy_orders_sorted:
            if price >= clearing_price and remain_buy > 0:
                exec_shares = min(shares, remain_buy)
                remain_buy -= exec_shares
                executed_buy_idx.add(idx)
                
        # Fill sell orders from lowest price up
        remain_sell = max_volume
        for idx, (price, shares, investor) in sell_orders_sorted:
            if price <= clearing_price and remain_sell > 0:
                exec_shares = min(shares, remain_sell)
                remain_sell -= exec_shares
                executed_sell_idx.add(idx)
                
        return clearing_price, max_volume, executed_buy_idx, executed_sell_idx

    def execute_trades(self, clearing_price, max_volume, buy_orders, sell_orders, executed_buy_idx, executed_sell_idx):
        """
        Execute trades at the clearing price and update investor positions
        
        Steps:
        1. Process buy orders:
           - Update investor shares and cash
           - Update value estimates for value investors
        2. Process sell orders:
           - Update investor shares and cash
           - Update value estimates for value investors
           
        Args:
            clearing_price: Price at which trades execute
            max_volume: Total volume to execute
            buy_orders: List of buy orders
            sell_orders: List of sell orders
            executed_buy_idx: Indices of buy orders to execute
            executed_sell_idx: Indices of sell orders to execute
        """
        if max_volume <= 0 or not buy_orders or not sell_orders or not executed_buy_idx or not executed_sell_idx:
            return
            
        # Process buy orders
        remain = max_volume
        buy_orders_sorted = sorted(enumerate(buy_orders), key=lambda x: x[1][0], reverse=True)
        if buy_orders_sorted:
            for idx, (price, shares, investor) in buy_orders_sorted:
                if idx in executed_buy_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares
                    investor.shares += exec_shares
                    investor.cash -= exec_shares * clearing_price
                    # Update value estimates for value investors
                    if hasattr(investor, 'value_estimate'):
                        error_factor = getattr(investor, 'estimation_error', 0.1)
                        investor.value_estimate = self.true_value + self._rng.normal(0, error_factor * max(self.true_value, 0.01))
        
        # Process sell orders
        remain = max_volume
        sell_orders_sorted = sorted(enumerate(sell_orders), key=lambda x: x[1][0])
        if sell_orders_sorted:
            for idx, (price, shares, investor) in sell_orders_sorted:
                if idx in executed_sell_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares
                    investor.shares -= exec_shares
                    investor.cash += exec_shares * clearing_price
                    # Update value estimates for value investors
                    if hasattr(investor, 'value_estimate'):
                        error_factor = getattr(investor, 'estimation_error', 0.1)
                        investor.value_estimate = self.true_value + self._rng.normal(0, error_factor * max(self.true_value, 0.01))

    def daily_auction(self):
        """
        Run the daily call auction process and update market state
        
        Steps:
        1. Run call auction to find clearing price
        2. Execute trades at clearing price
        3. Update price history and trading volume
        4. Clear order books
        5. Update true value (random jumps at intervals)
        """
        if not self.buy_orders or not self.sell_orders:
            self.price_history.append(self.price)
            self.executed_volume = 0
            self.executed_volume_history.append(0)
            self.value_history.append(self.true_value)
            return
            
        # Run auction
        last_price = self.price_history[-1]
        clearing_price, max_volume, executed_buy_idx, executed_sell_idx = self.call_auction(
            self.buy_orders, self.sell_orders, last_price
        )
        
        # Update market state
        self.price = clearing_price
        self.price_history.append(self.price)
        self.executed_volume = max_volume
        self.executed_volume_history.append(self.executed_volume)
        
        # Execute trades
        self.execute_trades(clearing_price, max_volume, self.buy_orders, self.sell_orders, 
                          executed_buy_idx, executed_sell_idx)
                          
        # Clear order books
        self.buy_orders = []
        self.sell_orders = []
        
        # Handle true value jumps
        current_day = len(self.price_history) - 1
        if self._next_jump_interval is None:
            self._next_jump_interval = self._rng_value.randint(30, 50)
            
        # Check if it's time for a value jump
        if current_day - self._last_jump_day >= self._next_jump_interval:
            if self._rng_value.rand() < 0.33:  # 33% chance of value jump
                change = self._rng_value.uniform(10, 30) * (1 if self._rng_value.rand() < 0.5 else -1)
                self.true_value += change
                if self.true_value < 0:
                    self.true_value = 0
            self._last_jump_day = current_day
            self._next_jump_interval = self._rng_value.randint(30, 50)
            
        self.value_history.append(self.true_value)

def simulate_stock_market():
    """
    Run a stock market simulation with multiple types of investors
    
    Simulation setup:
    1. Market parameters:
       - Initial stock price: 100
       - Minimum price tick: 0.01
       - Simulation duration: 8000 days
       - True value seed: 2106 for reproducibility
       
    2. Investor types and parameters:
       - Value Investors (100): Base decisions on estimated intrinsic value
       - Chase Investors (100): Follow price momentum
       - Trend Investors (100): Use moving average crossovers
       - Random Investors (100): Make random trading decisions
       - Never Stop Loss (100): Hold until profit target
       - Bottom Fishing (10): Buy on significant price drops
       
    3. Each investor starts with:
       - 100 shares of stock
       - 10,000 units of cash
       
    4. Visualization:
       - Price and volume chart
       - Average shares by investor type
       - Average cash by investor type
       - Average wealth by investor type
       - Trend investor performance by MA period
    """
    # Initialize market parameters
    initial_price = 100
    price_tick = 0.01
    days = 8000
    value_line_seed = 2106

    # Define investor group parameters
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
    
    never_stop_loss_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    bottom_fishing_investors_params = {
        'num': 10,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    # Create market instance with initial conditions
    market = Market(initial_price, price_tick, value_line_seed)

    investors = []
    trend_periods = [5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    trend_investors_by_period = {}

    # Initialize Value Investors
    for _ in range(value_investors_params['num']):
        value_estimate = market._rng.normal(market.true_value, 10)
        investors.append(ValueInvestor(
            value_investors_params['initial_shares'], 
            value_investors_params['initial_cash'], 
            value_estimate
        ))

    # Initialize Chase Investors
    for _ in range(chase_investors_params['num']):
        investors.append(ChaseInvestor(
            chase_investors_params['initial_shares'], 
            chase_investors_params['initial_cash']
        ))

    # Initialize Trend Investors across different periods
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

    # Initialize Random Investors
    for _ in range(random_investors_params['num']):
        investors.append(RandomInvestor(
            random_investors_params['initial_shares'], 
            random_investors_params['initial_cash']
        ))
        
    # Initialize Never Stop Loss Investors
    for _ in range(never_stop_loss_investors_params['num']):
        investor = NeverStopLossInvestor(
            never_stop_loss_investors_params['initial_shares'],
            never_stop_loss_investors_params['initial_cash'],
            buy_probability=0.2,
            profit_target=0.1
        )
        investors.append(investor)

    # Initialize Bottom Fishing Investors
    for _ in range(bottom_fishing_investors_params['num']):
        profit_target = np.random.uniform(0.1, 0.5)
        investors.append(BottomFishingInvestor(
            bottom_fishing_investors_params['initial_shares'],
            bottom_fishing_investors_params['initial_cash'],
            profit_target=profit_target
        ))

    # Calculate indices for different investor groups
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']

    # Initialize tracking variables
    prices = [initial_price]
    shares_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': []}
    cash_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': []}
    wealth_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': []}
    trend_assets_by_period = {period: [] for period in trend_periods}

    # Run simulation
    for _ in range(days):
        # Let each investor make trading decisions
        for investor in investors:
            investor.trade(market.price, market)
            
        # Execute daily market auction
        market.daily_auction()

        # Track performance metrics by investor type
        type_ranges = [
            ('Value', 0, value_end),
            ('Chase', value_end, chase_end),
            ('Trend', chase_end, trend_end),
            ('Random', trend_end, random_end),
            ('NeverStopLoss', random_end, never_stop_loss_end),
            ('BottomFishing', never_stop_loss_end, len(investors))
        ]

        # Calculate and record average metrics for each investor type
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
            
        # Track performance of trend investors by MA period
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

    # Create visualization subplots
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
    
    # Plot 1: Price, True Value, and Volume
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
    
    # Plot 2: Average Shares by Investor Type
    for type_name in shares_by_type:
        axs[1].plot(shares_by_type[type_name], label=f'{type_name} Investors')
    axs[1].set_ylabel('Shares')
    axs[1].set_title('Average Shares Held by Investor Type')
    axs[1].legend()
    
    # Plot 3: Average Cash by Investor Type
    for type_name in cash_by_type:
        axs[2].plot(cash_by_type[type_name], label=f'{type_name} Investors')
    axs[2].set_ylabel('Cash')
    axs[2].set_title('Average Cash Held by Investor Type')
    axs[2].legend()
    
    # Plot 4: Average Total Wealth by Investor Type
    for type_name in wealth_by_type:
        axs[3].plot(wealth_by_type[type_name], label=f'{type_name} Investors')
    axs[3].set_ylabel('Total Wealth')
    axs[3].set_title('Average Total Wealth by Investor Type')
    axs[3].legend()
    
    # Plot 5: Trend Investor Performance by MA Period
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
