# Version 3.1.0 Upgrade Notes (from v3.0.8)
# 
# Major Improvements:
# 1. New Investor Type:
#    - Added BottomFishingInvestor class implementing a "buy the dip" strategy
#    - Dynamically adjusts buying ratio based on price drop magnitude
#    - Uses customizable profit targets for selling decisions
#
# 2. Market Mechanism Improvements:
#    - Enhanced capital management with inject_capital() and withdraw_capital() methods
#    - Added support for both fixed amount and percentage-based capital changes
#    - Implemented capital_change_history to track all capital movements
#    - Improved trade execution and order matching in daily auctions
#
# 3. Simulation Enhancements:
#    - Added comprehensive capital event system for testing market dynamics
#    - Introduced new visualization markers for capital injection/withdrawal events
#    - Enhanced performance tracking for different investor types
#    - Improved trend analysis with multiple MA periods
#
# 4. Code Optimization:
#    - Better error handling in capital management functions
#    - Improved parameter validation for market operations
#    - Enhanced code documentation and structure
#    - More efficient data collection for analysis
# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

class Investor:
    def __init__(self, shares, cash):
        self.shares = shares
        self.cash = cash

    def trade(self, price, market):
        pass

    def decide_price(self, current_price, market):
        return ('hold', 0, 0)

class ValueInvestor(Investor):
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
    def __init__(self, shares, cash, N=None):
        super().__init__(shares, cash)
        self.N = N
        self._n_initialized = False

    def calculate_velocity(self, prices):
        if len(prices) < 2:
            return 0.0
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                change = (prices[i] - prices[i-1]) / prices[i-1]
                price_changes.append(change)
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

class NeverStopLossInvestor(Investor):
    def __init__(self, shares, cash, buy_probability=0.05, profit_target=0.1, seed=None):
        super().__init__(shares, cash)
        self.buy_price = None
        self.buy_probability = buy_probability
        self.profit_target = profit_target
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        if self.shares == 0 and self.buy_price is None:
            if self._rng_investor.random() < self.buy_probability:
                buy_shares = int(self.cash / current_price)
                if buy_shares > 0:
                    self.buy_price = current_price
                    buy_price = current_price * 1.01
                    return ('buy', buy_price, buy_shares)
        
        elif self.shares > 0 and self.buy_price is not None:
            profit_ratio = (current_price - self.buy_price) / self.buy_price
            
            if profit_ratio >= self.profit_target:
                sell_price = current_price * 0.99
                sell_shares = self.shares
                self.buy_price = None
                return ('sell', sell_price, sell_shares)
                
        return ('hold', 0, 0)

    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)

class BottomFishingInvestor(Investor):
    def __init__(self, shares, cash, profit_target=None, seed=None):
        super().__init__(shares, cash)
        self.avg_cost = None
        self.profit_target = profit_target if profit_target is not None else np.random.uniform(0.1, 0.5)
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.trigger_drop = self._rng_investor.uniform(0.05, 0.15)
        self.step_drop = self._rng_investor.uniform(0.05, 0.15)
        self.last_buy_price = None

    def decide_price(self, current_price, market):
        if self.shares > 0 and self.avg_cost is not None:
            profit_ratio = (current_price - self.avg_cost) / self.avg_cost
            if profit_ratio >= self.profit_target:
                sell_price = current_price * 0.99
                return ('sell', sell_price, self.shares)
                
        if len(market.price_history) >= 100:
            peak_price = max(market.price_history[-100:])
            drop_from_peak = (peak_price - current_price) / peak_price
            
            if drop_from_peak >= self.trigger_drop:
                drop_steps = int((drop_from_peak - self.trigger_drop) / self.step_drop)
                if drop_steps > 0:
                    buy_ratio = min(0.8, 0.1 + drop_steps * 0.1)
                    buy_shares = int((self.cash * buy_ratio) / current_price)
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        return ('buy', buy_price, buy_shares)
                        
        return ('hold', 0, 0)

    def trade(self, price, market):
        action, order_price, shares = self.decide_price(price, market)
        if action == 'buy':
            total_cost = (self.avg_cost * self.shares if self.avg_cost is not None else 0) + order_price * shares
            total_shares = self.shares + shares
            self.avg_cost = total_cost / total_shares if total_shares > 0 else None
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)
            self.avg_cost = None

class Market:
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
        self.capital_change_history = []

    def place_order(self, order_type, price, shares, investor):
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

    def inject_capital(self, investors, amount=None, percentage=None, day=None):
        if amount is None and percentage is None:
            raise ValueError("Must provide either amount or percentage parameter")
        if amount is not None and percentage is not None:
            raise ValueError("Cannot provide both amount and percentage parameters")
            
        current_day = day if day is not None else len(self.price_history) - 1
        
        for investor in investors:
            if amount is not None:
                investor.cash += amount
                injection_amount = amount
            else:
                injection_amount = investor.cash * percentage
                investor.cash += injection_amount
                
        if amount is not None:
            self.capital_change_history.append((current_day, "inject", amount, None))
        else:
            self.capital_change_history.append((current_day, "inject", None, percentage))
    
    def withdraw_capital(self, investors, amount=None, percentage=None, day=None):
        if amount is None and percentage is None:
            raise ValueError("Must provide either amount or percentage parameter")
        if amount is not None and percentage is not None:
            raise ValueError("Cannot provide both amount and percentage parameters")
            
        current_day = day if day is not None else len(self.price_history) - 1
        
        for investor in investors:
            if amount is not None:
                withdrawal_amount = min(amount, investor.cash)
                investor.cash -= withdrawal_amount
            else:
                withdrawal_amount = investor.cash * percentage
                investor.cash -= withdrawal_amount
                
        if amount is not None:
            self.capital_change_history.append((current_day, "withdraw", amount, None))
        else:
            self.capital_change_history.append((current_day, "withdraw", None, percentage))
    
    def daily_auction(self):
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
    days = 1000
    value_line_seed = 2073

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
        
    for _ in range(never_stop_loss_investors_params['num']):
        investor = NeverStopLossInvestor(
            never_stop_loss_investors_params['initial_shares'],
            never_stop_loss_investors_params['initial_cash'],
            buy_probability=0.2,
            profit_target=0.1
        )
        investors.append(investor)

    for _ in range(bottom_fishing_investors_params['num']):
        profit_target = np.random.uniform(0.1, 0.5)
        investors.append(BottomFishingInvestor(
            bottom_fishing_investors_params['initial_shares'],
            bottom_fishing_investors_params['initial_cash'],
            profit_target=profit_target
        ))

    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']

    prices = [initial_price]
    shares_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': []}
    cash_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': []}
    wealth_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': []}
    trend_assets_by_period = {period: [] for period in trend_periods}

    capital_events = [
        {'day': 300, 'type': 'withdraw', 'percentage': 0.6, 'investors_range': (chase_end, trend_end)},
    ]
    
    for day in range(days):
        for event in capital_events:
            if day == event['day']:
                start, end = event['investors_range']
                affected_investors = investors[start:end]
                if event['type'] == 'inject':
                    if 'amount' in event:
                        market.inject_capital(affected_investors, amount=event['amount'], day=day)
                    else:
                        market.inject_capital(affected_investors, percentage=event['percentage'], day=day)
                else:
                    if 'amount' in event:
                        market.withdraw_capital(affected_investors, amount=event['amount'], day=day)
                    else:
                        market.withdraw_capital(affected_investors, percentage=event['percentage'], day=day)
        
        for investor in investors:
            investor.trade(market.price, market)
        market.daily_auction()

        type_ranges = [
            ('Value', 0, value_end),
            ('Chase', value_end, chase_end),
            ('Trend', chase_end, trend_end),
            ('Random', trend_end, random_end),
            ('NeverStopLoss', random_end, never_stop_loss_end),
            ('BottomFishing', never_stop_loss_end, len(investors))
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
    
    if market.capital_change_history:
        for day, event_type, amount, percentage in market.capital_change_history:
            if event_type == 'inject':
                color = 'green'
                marker = '^'
                label = f'Inject: {amount if amount else percentage*100:.1f}%'
            else:
                color = 'red'
                marker = 'v'
                label = f'Withdraw: {amount if amount else percentage*100:.1f}%'
            
            axs[0].axvline(x=day, color=color, alpha=0.3, linestyle='--')
            axs[0].plot(day, market.price_history[day], marker=marker, markersize=10, color=color, label=label)
    
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
    
    if market.capital_change_history:
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_stock_market()
