# Version 3.2.1.1
# Upgrade notes from version 3.1.0:
# 1. Enhanced investor behavior modeling:
#    - Added Bottom Fishing Investor class for implementing batch buying strategy
#    - Added Message Investor class with delayed information processing
#    - Removed Insider Trader class to maintain market fairness
#
# 2. Improved market mechanism:
#    - Added transaction fee mechanism (buy_fee_rate and sell_fee_rate)
#    - Implemented capital injection and withdrawal functionality
#    - Added executed volume tracking and history
#
# 3. Enhanced visualization:
#    - Added trading volume display in the first subplot
#    - Added capital change event markers on price chart
#    - Improved trend investor performance analysis by MA periods
#
# 4. Code optimization:
#    - Improved error handling and parameter validation
#    - Enhanced documentation and code comments
#    - Better separation of concerns in market operations

import numpy as np
import matplotlib.pyplot as plt

class Investor:
    """
    Base Investor class - Defines basic properties and methods for all investors

    Properties:
        shares: Number of shares held
        cash: Cash held
    """
    def __init__(self, shares, cash):
        self.shares = shares  # shares held
        self.cash = cash      # cash held

    def trade(self, price, market):
        """Trading method to be implemented by subclasses"""
        pass

    def decide_price(self, current_price, market):
        """Method to decide trading price and quantity, implemented by subclasses
        Returns: (action, price, shares)
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
    - Trading volume proportional to price deviation

    Properties:
        value_estimate: Estimated value of the stock
        k: Trading sensitivity coefficient
    """
    def __init__(self, shares, cash, value_estimate, k=1, estimation_error=0.1):
        super().__init__(shares, cash)
        self.value_estimate = value_estimate  # estimated stock value
        self.k = k  # trading sensitivity
        self.estimation_error = estimation_error  # standard deviation of estimation error

    def decide_price(self, current_price, market):
        """Decide trading action based on difference between current price and estimated value"""
        diff = (current_price - self.value_estimate) / self.value_estimate  # Calculate price deviation percentage
        if diff > 0:  # Current price above estimate, consider selling
            sell_amount = self.k * diff * self.value_estimate  # Selling amount proportional to price deviation
            sell_shares = min(sell_amount, self.shares)
            if sell_shares > 0:
                sell_price = max(current_price * 0.99, self.value_estimate)  # Selling price slightly below market
                return ('sell', sell_price, sell_shares)
        elif diff < 0:  # Current price below estimate, consider buying
            buy_amount = self.k * -diff * self.value_estimate  # Buying amount proportional to price deviation
            buy_shares = min(buy_amount, self.cash / current_price)
            if buy_shares > 0:
                buy_price = min(current_price * 1.01, self.value_estimate)  # Buying price slightly above market
                return ('buy', buy_price, buy_shares)
        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        """Execute specific trading operation"""
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class ChaseInvestor(Investor):
    """
    Chase Investor - Trades based on price change velocity

    Strategy:
    - Buy more when price rises faster (proportional to remaining cash)
    - Sell more when price falls faster (proportional to remaining shares)
    - Trading volume proportional to price change velocity

    Properties:
        N: Observation period for calculating price change velocity
    """
    def __init__(self, shares, cash, N=None):
        super().__init__(shares, cash)
        self.N = N  # observation period
        self._n_initialized = False  # flag to track if N is initialized

    def calculate_velocity(self, prices):
        """Calculate price change velocity"""
        if len(prices) < 2:
            return 0.0
        # Calculate price change rate
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:  # ensure denominator is positive, avoid division by zero or negative
                change = (prices[i] - prices[i-1]) / prices[i-1]
                price_changes.append(change)
        # Check if there are valid price changes
        if not price_changes:
            return 0.0
        # Return average change rate
        return float(sum(price_changes)) / len(price_changes) if price_changes else 0.0  # ensure float return and check list is not empty

    def decide_price(self, current_price, market):
        """Decide trading action based on price change velocity"""
        # Ensure N is initialized
        if self.N is None and not self._n_initialized:
            self.N = market._rng.choice([3, 5, 10, 15, 20])
            self._n_initialized = True

        if len(market.price_history) >= self.N:
            recent_prices = market.price_history[-self.N:]  # get recent price history
            velocity = self.calculate_velocity(recent_prices)  # calculate price change velocity

            if velocity > 0:  # Price rising trend
                # Buy ratio proportional to velocity, max 80% of cash
                buy_ratio = min(abs(velocity) * 5, 1)  # Convert velocity to buy ratio
                buy_shares = int((self.cash * buy_ratio) / current_price)
                if buy_shares > 0:
                    buy_price = current_price * 1.02  # accept slightly higher buying price
                    return ('buy', buy_price, buy_shares)
            elif velocity < 0:  # Price falling trend
                # Sell ratio proportional to velocity, max 80% of shares
                sell_ratio = min(abs(velocity) * 5, 1)  # Convert velocity to sell ratio
                sell_shares = int(self.shares * sell_ratio)
                if sell_shares > 0:
                    sell_price = current_price * 0.98  # accept slightly lower selling price
                    return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)  # no trading signal

    def trade(self, price, market):
        """Execute specific trading operation"""
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

    Properties:
        M: Moving average period
        above_ma: Record if price is above MA
    """
    def __init__(self, shares, cash, M):
        super().__init__(shares, cash)
        self.M = M  # moving average period
        self.above_ma = None  # Record if price is above MA

    def decide_price(self, current_price, market):
        """Decide trading action based on price's relationship with moving average"""
        if len(market.price_history) >= self.M:
            # calculate simple moving average
            recent_prices = market.price_history[-self.M:]
            if not recent_prices or len(recent_prices) < self.M:  # check if enough price data
                return ('hold', 0, 0)
            sma = float(sum(recent_prices)) / len(recent_prices)  # already checked list is not empty, safe to use length
            current_above_ma = current_price > sma

            # detect price crossing MA
            if self.above_ma is None:
                self.above_ma = current_above_ma
            elif current_above_ma != self.above_ma:  # price crosses MA
                self.above_ma = current_above_ma
                if current_above_ma:  # crosses above MA, buy all
                    buy_shares = self.cash // current_price  # calculate maximum shares to buy
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        return ('buy', buy_price, buy_shares)
                else:  # crosses below MA, sell all
                    if self.shares > 0:
                        sell_price = current_price * 0.99
                        return ('sell', sell_price, self.shares)
        return ('hold', 0, 0)  # no trading signal

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

    Properties:
        p: Probability of buying or selling
        ratio: Fixed ratio for each trade
    """
    def __init__(self, shares, cash, p=0.2, ratio=0.1, seed=None):
        super().__init__(shares, cash)
        self.p = p  # trading probability
        self.ratio = ratio  # fixed trading ratio
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        """Randomly decide trading action and price"""
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
        """Execute specific trading operation"""
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class NeverStopLossInvestor(Investor):
    """
    Never Stop Loss Investor - Holds positions until price recovers to entry point

    Strategy:
    - May buy all-in at random moments
    - If in loss after buying, continues to hold
    - Only considers selling when price rises above entry point

    Properties:
        buy_price: Entry price, None if not holding
        buy_probability: Probability of buying
        profit_target: Target profit percentage
    """
    def __init__(self, shares, cash, buy_probability=0.05, profit_target=0.1, seed=None):
        super().__init__(shares, cash)
        self.buy_price = None  # Entry price, initially None
        self.buy_probability = buy_probability  # Probability of buying
        self.profit_target = profit_target  # Target profit percentage
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        """Decide trading action based on current position status and price"""
        # If no position, consider buying
        if self.shares == 0 and self.buy_price is None:
            # Randomly decide whether to buy
            if self._rng_investor.random() < self.buy_probability:
                # Buy all-in
                buy_shares = int(self.cash / current_price)
                if buy_shares > 0:
                    self.buy_price = current_price  # Record entry price
                    buy_price = current_price * 1.01  # Accept slightly higher buy price
                    return ('buy', buy_price, buy_shares)

        # If holding position and current price is above entry price, consider selling
        elif self.shares > 0 and self.buy_price is not None:
            # Calculate current profit ratio
            profit_ratio = (current_price - self.buy_price) / self.buy_price

            # If profit target reached, sell
            if profit_ratio >= self.profit_target:
                sell_price = current_price * 0.99  # Accept slightly lower sell price
                sell_shares = self.shares  # Sell all shares
                self.buy_price = None  # Reset entry price
                return ('sell', sell_price, sell_shares)

        return ('hold', 0, 0)  # No trading signal or holding waiting for recovery

    def trade(self, price, market):
        """Execute specific trading operation"""
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class BottomFishingInvestor(Investor):
    """
    Bottom Fishing Investor - Buys in batches after price drops x% from 100-day high

    Properties:
        profit_target: Target profit percentage (10%-50%)
        avg_cost: Weighted average cost of holding positions
        trigger_drop: Trigger drop percentage for buying (5%-15%)
        step_drop: Drop percentage for each batch buy (5%-15%)
    """
    def __init__(self, shares, cash, profit_target=None, seed=None):
        super().__init__(shares, cash)
        self.avg_cost = None  # Weighted average cost of positions
        self.profit_target = profit_target if profit_target is not None else np.random.uniform(0.1, 0.5)
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.trigger_drop = self._rng_investor.uniform(0.05, 0.15)  # Trigger percentage for buying
        self.step_drop = self._rng_investor.uniform(0.05, 0.15)    # Percentage for each batch buy
        self.last_buy_price = None  # Record last buying price

    def decide_price(self, current_price, market):
        # Selling logic: sell all when profit target is reached
        if self.shares > 0 and self.avg_cost is not None:
            profit_ratio = (current_price - self.avg_cost) / self.avg_cost
            if profit_ratio >= self.profit_target:
                sell_price = current_price * 0.99
                return ('sell', sell_price, self.shares)

        # Buying logic: start buying when price drops trigger_drop% from 100-day high, buy more every step_drop%
        if len(market.price_history) >= 100:
            peak_price = max(market.price_history[-100:])
            drop_from_peak = (peak_price - current_price) / peak_price

            # Check if drop percentage reaches trigger point
            if drop_from_peak >= self.trigger_drop:
                # Calculate how many steps price has dropped
                drop_steps = int((drop_from_peak - self.trigger_drop) / self.step_drop)
                if drop_steps > 0:
                    # Buy more as price drops more (max 80% of cash)
                    buy_ratio = min(0.8, 0.1 + drop_steps * 0.1)
                    buy_shares = int((self.cash * buy_ratio) / current_price)
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        return ('buy', buy_price, buy_shares)

        return ('hold', 0, 0)

    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            # Update weighted average cost
            total_cost = (self.avg_cost * self.shares if self.avg_cost is not None else 0) + order_price * shares
            total_shares = self.shares + shares
            self.avg_cost = total_cost / total_shares if total_shares > 0 else None
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)
            self.avg_cost = None  # Reset cost after selling all positions

# 内幕交易者类已移除

class MessageInvestor(Investor):
    """
    Message Investor - Receives information about stock value changes with 1-5 days delay

    Properties:
        delay_days: Days of delay before receiving value change information (1-5 days)
        profit_target: Target profit percentage (default 15%)
        stop_loss: Stop loss percentage (default 20%)
        max_hold_days: Maximum holding period (default 30 days)
        holding_days: Current holding period
        entry_price: Entry price
    """
    def __init__(self, shares, cash, delay_days=None, profit_target=0.39, stop_loss=0.20, max_hold_days=200, seed=None):
        super().__init__(shares, cash)
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.delay_days = delay_days if delay_days is not None else self._rng.randint(1, 6)
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
        self.holding_days = 0
        self.entry_price = None

    def decide_price(self, current_price, market):
        # Update holding period
        if self.shares > 0 and self.entry_price is not None:
            self.holding_days += 1
            profit_ratio = (current_price - self.entry_price) / self.entry_price
            if (profit_ratio >= self.profit_target or
                profit_ratio <= -self.stop_loss or
                self.holding_days >= self.max_hold_days):
                sell_price = current_price * 0.99
                sell_shares = self.shares
                self.entry_price = None
                self.holding_days = 0
                return ('sell', sell_price, sell_shares)

        # Get delayed value change information
        current_day = len(market.price_history) - 1
        if current_day >= self.delay_days:  # Ensure enough historical data
            past_value = market.value_history[current_day - self.delay_days]
            current_value = market.value_history[current_day]
            value_change = current_value - past_value

            if abs(value_change) > 0:  # If value change detected
                if value_change > 0:  # Value increases and no position, buy
                    buy_shares = int(self.cash * 0.99 / current_price)
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        self.entry_price = current_price
                        self.holding_days = 0
                        return ('buy', buy_price, buy_shares)
                elif value_change < 0 and self.shares > 0:  # Value decreases and holding position, sell
                    sell_price = current_price * 0.99
                    sell_shares = self.shares
                    self.entry_price = None
                    self.holding_days = 0
                    return ('sell', sell_price, sell_shares)

        return ('hold', 0, 0)

    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action in ['buy', 'sell']:
            market.place_order(action, order_price, shares, self)

class Market:
    """
    Stock Market - Market simulator implementing call auction mechanism

    Strategy:
    - Collect buy and sell orders
    - Determine clearing price through call auction
    - Execute trades based on price priority
    - Stock value changes as a random square wave
    - Support capital injection and withdrawal on specified dates
    - Implement transaction fee mechanism

    Properties:
        price: Current market price
        price_tick: Minimum price movement unit
        price_history: Price history
        buy_orders: Buy order queue
        sell_orders: Sell order queue
        executed_volume: Executed trading volume
        true_value: True stock value
        value_history: Value history
        seed: Random seed
        capital_change_history: Capital change history
        buy_fee_rate: Buy transaction fee rate
        sell_fee_rate: Sell transaction fee rate
        fee_income: Accumulated fee income
    """
    def __init__(self, initial_price, price_tick=0.01, seed=None, buy_fee_rate=0.001, sell_fee_rate=0.002):
        self.price = initial_price  # Initial price
        self.price_tick = price_tick  # Minimum price tick
        self.price_history = [initial_price]  # Price history
        self.buy_orders = []  # Buy orders queue
        self.sell_orders = []  # Sell orders queue
        self.executed_volume = 0  # Executed volume
        self.true_value = initial_price  # Initial true value
        self.value_history = [initial_price]  # Value history
        self._last_jump_day = 0  # Last day when value jumped
        self._next_jump_interval = None  # Interval to next jump
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self._rng_value = np.random.RandomState(self.seed)
        self._rng = np.random.RandomState(self.seed + 9999)
        self.executed_volume_history = []  # Daily executed volume history
        self.capital_change_history = []  # Capital injection/withdrawal history [(day, type, amount, percentage)]
        self.buy_fee_rate = buy_fee_rate  # Buy transaction fee rate, default 0.1%
        self.sell_fee_rate = sell_fee_rate  # Sell transaction fee rate, default 0.2%
        self.fee_income = 0  # Accumulated fee income

    def place_order(self, order_type, price, shares, investor):
        if order_type == 'buy':
            self.buy_orders.append((price, shares, investor))
        elif order_type == 'sell':
            self.sell_orders.append((price, shares, investor))

    def call_auction(self, buy_orders, sell_orders, last_price):
        # 检查订单列表是否为空
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
        # Ensure all parameters are valid
        if max_volume <= 0 or not buy_orders or not sell_orders or not executed_buy_idx or not executed_sell_idx:
            return

        # Execute buy orders
        remain = max_volume
        buy_orders_sorted = sorted(enumerate(buy_orders), key=lambda x: x[1][0], reverse=True)
        if buy_orders_sorted:  # Ensure there are buy orders
            for idx, (price, shares, investor) in buy_orders_sorted:
                if idx in executed_buy_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares

                    # Calculate transaction amount and buy fee
                    trade_amount = exec_shares * clearing_price
                    buy_fee = trade_amount * self.buy_fee_rate

                    # Update investor's position and cash (deduct fee)
                    investor.shares += exec_shares
                    investor.cash -= (trade_amount + buy_fee)

                    # Accumulate fee income
                    self.fee_income += buy_fee

                    if hasattr(investor, 'value_estimate'):
                        # Update valuation safely
                        error_factor = getattr(investor, 'estimation_error', 0.1)
                        investor.value_estimate = self.true_value + self._rng.normal(0, error_factor * max(self.true_value, 0.01))

        # Execute sell orders
        remain = max_volume
        sell_orders_sorted = sorted(enumerate(sell_orders), key=lambda x: x[1][0])
        if sell_orders_sorted:  # Ensure there are sell orders
            for idx, (price, shares, investor) in sell_orders_sorted:
                if idx in executed_sell_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares

                    # Calculate transaction amount and sell fee
                    trade_amount = exec_shares * clearing_price
                    sell_fee = trade_amount * self.sell_fee_rate

                    # Update investor's position and cash (deduct fee)
                    investor.shares -= exec_shares
                    investor.cash += (trade_amount - sell_fee)

                    # Accumulate fee income
                    self.fee_income += sell_fee

                    if hasattr(investor, 'value_estimate'):
                        # Update valuation safely
                        error_factor = getattr(investor, 'estimation_error', 0.1)
                        investor.value_estimate = self.true_value + self._rng.normal(0, error_factor * max(self.true_value, 0.01))

    def inject_capital(self, investors, amount=None, percentage=None, day=None):
        """
        Inject capital to investors

        Args:
            investors (list): List of investors
            amount (float, optional): Fixed amount to inject
            percentage (float, optional): Percentage of current cash to inject
            day (int, optional): Day of injection, default is current day

        Note: Either amount or percentage must be provided, but not both
        """
        if amount is None and percentage is None:
            raise ValueError("Must provide either amount or percentage parameter")
        if amount is not None and percentage is not None:
            raise ValueError("Cannot provide both amount and percentage parameters")

        # Determine current day
        current_day = day if day is not None else len(self.price_history) - 1

        # Inject capital for each investor
        for investor in investors:
            if amount is not None:
                # Inject fixed amount
                investor.cash += amount
                injection_amount = amount
            else:
                # Inject by percentage
                injection_amount = investor.cash * percentage
                investor.cash += injection_amount

        # Record capital injection history
        if amount is not None:
            self.capital_change_history.append((current_day, "inject", amount, None))
        else:
            self.capital_change_history.append((current_day, "inject", None, percentage))

    def withdraw_capital(self, investors, amount=None, percentage=None, day=None):
        """
        Withdraw capital from investors

        Args:
            investors (list): List of investors
            amount (float, optional): Fixed amount to withdraw
            percentage (float, optional): Percentage of current cash to withdraw
            day (int, optional): Day of withdrawal, default is current day

        Note: Either amount or percentage must be provided, but not both
        """
        if amount is None and percentage is None:
            raise ValueError("Must provide either amount or percentage parameter")
        if amount is not None and percentage is not None:
            raise ValueError("Cannot provide both amount and percentage parameters")

        # Determine current day
        current_day = day if day is not None else len(self.price_history) - 1

        # Withdraw capital from each investor
        for investor in investors:
            if amount is not None:
                # Withdraw fixed amount, but not more than available cash
                withdrawal_amount = min(amount, investor.cash)
                investor.cash -= withdrawal_amount
            else:
                # Withdraw by percentage
                withdrawal_amount = investor.cash * percentage
                investor.cash -= withdrawal_amount

        # Record capital withdrawal history
        if amount is not None:
            self.capital_change_history.append((current_day, "withdraw", amount, None))
        else:
            self.capital_change_history.append((current_day, "withdraw", None, percentage))

    def daily_auction(self):
        # One auction per day
        if not self.buy_orders or not self.sell_orders:
            # If no orders, still need to update price history and executed volume history
            self.price_history.append(self.price)
            self.executed_volume = 0
            self.executed_volume_history.append(0)
            # Update true value history
            self.value_history.append(self.true_value)
            return
        last_price = self.price_history[-1]
        clearing_price, max_volume, executed_buy_idx, executed_sell_idx = self.call_auction(self.buy_orders, self.sell_orders, last_price)
        self.price = clearing_price
        self.price_history.append(self.price)
        self.executed_volume = max_volume
        self.executed_volume_history.append(self.executed_volume) # Record executed volume history
        self.execute_trades(clearing_price, max_volume, self.buy_orders, self.sell_orders, executed_buy_idx, executed_sell_idx)
        self.buy_orders = []
        self.sell_orders = []
        # Random square wave changes in true value
        current_day = len(self.price_history) - 1  # Fix current day calculation
        if self._next_jump_interval is None:
            self._next_jump_interval = self._rng_value.randint(30, 50)
        if current_day - self._last_jump_day >= self._next_jump_interval:
            if self._rng_value.rand() < 0.33:
                change = self._rng_value.uniform(10, 30) * (1 if self._rng_value.rand() < 0.5 else -1)
                self.true_value += change

            self._last_jump_day = current_day
            self._next_jump_interval = self._rng_value.randint(30, 50)
        self.value_history.append(self.true_value)

    def get_future_value_change(self, prediction_days):
        current_day = len(self.price_history) - 1
        days_since_last_jump = current_day - self._last_jump_day
        if self._next_jump_interval is not None and days_since_last_jump + prediction_days >= self._next_jump_interval:
            if self._rng_value.rand() < 0.33:
                change = self._rng_value.uniform(10, 30) * (1 if self._rng_value.rand() < 0.5 else -1)
                return change
        return None

def simulate_stock_market():
    # Basic market parameter settings
    initial_price = 100
    price_tick = 0.01
    days = 2000
    value_line_seed = 2107
    buy_fee_rate = 0.01
    sell_fee_rate = 0.01

    # Parameters for different types of investors
    value_investors_params = {
        'num': 100,  # Large number of value investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    chase_investors_params = {
        'num': 100,  # Moderate number of momentum investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    trend_investors_params = {
        'num': 100,  # Relatively fewer trend investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    random_investors_params = {
        'num': 100,  # Large number of random investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    never_stop_loss_investors_params = {
        'num': 100,  # Moderate number of never-stop-loss investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    bottom_fishing_investors_params = {
        'num': 0,  # Few bottom fishing investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    insider_investors_params = {
        'num': 0,  # Remove insider traders
        'initial_shares': 100,
        'initial_cash': 10000
    }

    message_investors_params = {
        'num': 20,  # Few message investors but more than insiders
        'initial_shares': 100,
        'initial_cash': 10000
    }

    market = Market(initial_price, price_tick, value_line_seed, buy_fee_rate, sell_fee_rate)

    investors = []
    trend_periods = [5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    trend_investors_by_period = {}

    # Create value investors
    for _ in range(value_investors_params['num']):
        value_estimate = market._rng.normal(market.true_value, 10)
        investors.append(ValueInvestor(
            value_investors_params['initial_shares'],
            value_investors_params['initial_cash'],
            value_estimate
        ))

    # Create chase (momentum) investors
    for _ in range(chase_investors_params['num']):
        investors.append(ChaseInvestor(
            chase_investors_params['initial_shares'],
            chase_investors_params['initial_cash']
        ))

    # Create trend investors
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

    # Create random investors
    for _ in range(random_investors_params['num']):
        investors.append(RandomInvestor(
            random_investors_params['initial_shares'],
            random_investors_params['initial_cash']
        ))

    # Create never-stop-loss investors
    for _ in range(never_stop_loss_investors_params['num']):
        investor = NeverStopLossInvestor(
            never_stop_loss_investors_params['initial_shares'],
            never_stop_loss_investors_params['initial_cash'],
            buy_probability=0.2,  # 20% probability to buy all-in
            profit_target=0.1  # 10% target profit ratio
        )
        investors.append(investor)

    # Create bottom fishing investors
    for _ in range(bottom_fishing_investors_params['num']):
        profit_target = np.random.uniform(0.1, 0.5)
        investors.append(BottomFishingInvestor(
            bottom_fishing_investors_params['initial_shares'],
            bottom_fishing_investors_params['initial_cash'],
            profit_target=profit_target
        ))

    # Create message investors
    for _ in range(message_investors_params['num']):
        investors.append(MessageInvestor(
            message_investors_params['initial_shares'],
            message_investors_params['initial_cash']
        ))

    # Calculate start and end indices for each type of investor
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']
    bottom_fishing_end = never_stop_loss_end + bottom_fishing_investors_params['num']
    insider_end = bottom_fishing_end + insider_investors_params['num']  # Insider traders count is 0, but keep index calculation
    message_end = insider_end + message_investors_params['num']

    shares_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': [], 'Message': []}
    cash_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': [], 'Message': []}
    wealth_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': [], 'Message': []}
    trend_assets_by_period = {period: [] for period in trend_periods}

    # Define dates and parameters for capital injection and withdrawal
    # capital_events = [
    #     {'day': 500, 'type': 'inject', 'amount': 10000, 'investors_range': (0, len(investors))},  # Day 1000: Inject 5000 to all investors
    #     {'day': 1000, 'type': 'inject', 'percentage': 0.8, 'investors_range': (0, value_end)},  # Day 2000: Inject 20% to value investors
    #     {'day': 2000, 'type': 'withdraw', 'percentage': 0.5, 'investors_range': (0, len(investors))},  # Day 3000: Withdraw 10% from all investors
    #     {'day': 2500, 'type': 'withdraw', 'amount': 2000, 'investors_range': (random_end, never_stop_loss_end)}  # Day 4000: Withdraw 2000 from never-stop-loss investors
    # ]

    # capital_events = [
    #     {'day': 400, 'type': 'withdraw', 'percentage': 0.7, 'investors_range': (0, len(investors))},  # Day 2000: Inject 20% to value investors
    # ]

    capital_events = []

    for day in range(days):
        # Check for capital injection or withdrawal events
        for event in capital_events:
            if day == event['day']:
                start, end = event['investors_range']
                affected_investors = investors[start:end]
                if event['type'] == 'inject':
                    if 'amount' in event:
                        market.inject_capital(affected_investors, amount=event['amount'], day=day)
                    else:
                        market.inject_capital(affected_investors, percentage=event['percentage'], day=day)
                else:  # withdraw
                    if 'amount' in event:
                        market.withdraw_capital(affected_investors, amount=event['amount'], day=day)
                    else:
                        market.withdraw_capital(affected_investors, percentage=event['percentage'], day=day)

        for investor in investors:
            investor.trade(market.price, market)
        # Execute daily call auction
        market.daily_auction()
        # No need to append price again as daily_auction has already done it

        # Record average holdings, cash and total wealth for each type of investor
        type_ranges = [
            ('Value', 0, value_end),
            ('Chase', value_end, chase_end),
            ('Trend', chase_end, trend_end),
            ('Random', trend_end, random_end),
            ('NeverStopLoss', random_end, never_stop_loss_end),
            ('BottomFishing', never_stop_loss_end, bottom_fishing_end),
            # Insider traders removed but keep index calculation
            ('Message', insider_end, message_end)
        ]

        for type_name, start, end in type_ranges:
            # Only calculate averages if there are investors
            if start < end:
                shares_list = [inv.shares for inv in investors[start:end]]
                cash_list = [inv.cash for inv in investors[start:end]]
                if shares_list and cash_list:  # Ensure lists are not empty
                    avg_shares = np.mean(shares_list)  # Calculate average shares held
                    avg_cash = np.mean(cash_list)      # Calculate average cash
                    avg_wealth = avg_cash + avg_shares * market.price  # Calculate average total wealth
                else:
                    avg_shares = avg_cash = avg_wealth = 0.0
            else:
                avg_shares = avg_cash = avg_wealth = 0.0
            shares_by_type[type_name].append(avg_shares)
            cash_by_type[type_name].append(avg_cash)
            wealth_by_type[type_name].append(avg_wealth)

        # Record total assets for trend investors with different periods
        for period in trend_periods:
            investors_list = trend_investors_by_period[period]
            if investors_list:  # Ensure there are investors
                assets_list = [inv.cash + inv.shares * market.price for inv in investors_list]
                if assets_list:  # Ensure list is not empty
                    avg_assets = np.mean(assets_list)
                else:
                    avg_assets = 0.0
            else:
                avg_assets = 0.0
            trend_assets_by_period[period].append(avg_assets)

    # Create plots to display simulation results
    # Create visualization - 5 subplots
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)  # Share x-axis

    # Plot stock price and true value (first subplot)
    axs[0].plot(market.price_history, label='Stock Price', color='blue')  # Stock price
    axs[0].plot(market.value_history, label='True Value', linestyle='--', color='green')  # True value
    axs[0].set_ylabel('Price', color='blue')  # Set y-axis label
    axs[0].tick_params(axis='y', labelcolor='blue')  # Set tick color
    axs[0].legend(loc='upper left')  # Add legend

    # Add trading volume (second y-axis) to first subplot
    ax2 = axs[0].twinx()  # Create twin y-axis sharing x-axis
    # Ensure trading volume history matches price history length
    # Ensure plotted volume data matches days length
    daily_volumes = market.executed_volume_history
    ax2.bar(range(len(daily_volumes)), daily_volumes, alpha=0.3, color='gray', label='Trading Volume')  # Volume bar chart
    ax2.set_ylabel('Volume', color='gray')  # Set y-axis label
    ax2.tick_params(axis='y', labelcolor='gray')  # Set tick color
    ax2.legend(loc='upper right')  # Add legend
    axs[0].set_title('Stock Price, True Value, and Trading Volume Over Time')  # Set title

    # Add capital injection and withdrawal event markers to first subplot
    if market.capital_change_history:
        for day, event_type, amount, percentage in market.capital_change_history:
            if event_type == 'inject':
                color = 'green'  # Green for capital injection
                marker = '^'     # Up triangle for injection
                label = f'Inject: {amount if amount else percentage*100:.1f}%'
            else:  # withdraw
                color = 'red'    # Red for capital withdrawal
                marker = 'v'     # Down triangle for withdrawal
                label = f'Withdraw: {amount if amount else percentage*100:.1f}%'

            # Mark the event on the plot
            axs[0].axvline(x=day, color=color, alpha=0.3, linestyle='--')
            axs[0].plot(day, market.price_history[day], marker=marker, markersize=10, color=color, label=label)

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

    # Set common x-label for the fifth subplot
    axs[4].set_xlabel('Day')

    # Add legend for capital change events in the first subplot
    if market.capital_change_history:
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(), loc='upper left')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Run simulation
if __name__ == "__main__":
    simulate_stock_market()