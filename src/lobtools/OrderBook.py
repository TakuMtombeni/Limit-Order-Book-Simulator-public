# %% Dependencies
from collections import deque
from matplotlib import pyplot
from numba import jit

# %% OrderBook Data Structure Implementation
class OrderBook:
    """
    Provides an OrderBook object that allows for submission of limit and market
    orders, the cancellation of limit orders and matches incoming orders with
    active orders. Price-Time Priority is used for matching. No hidden orders.

    :param : (optional) A string to be used to identify the LOB
    :param : the tick size of the lob
    :param : the lot size of the lob
    """

    def __init__(self, ticker=None, tick_size=1, lot_size=1):
        self.ticker = ticker
        self.tick_size = tick_size
        self.lot_size = lot_size

        # dictionary to store buy limit orders, key = price_level, value = queue of limit orders
        self.buy_side = {}
        self.sell_side = {}

        self.next_orderid = 0  # variable to track next available order id
        self.time = 0  # variable to track time in the order book (in number of seconds from open)

        self.limit_orders = {}  # stores all limit orders submitted to the order book and their current status
        self.last_market_order_direction = 0  # stores direction of last market order

    def __getitem__(self, index):
        return self.limit_orders[index]

    def bestbid(self):
        """Returns the best bid"""
        return max(self.buy_side)

    def bestask(self):
        """Returns the best ask"""
        return min(self.sell_side)

    def spread(self):
        """Returns the current spread"""
        return self.bestask() - self.bestbid()

    def mid_price(self):
        """Returns the mid-price"""
        return 0.5 * (self.bestask() + self.bestbid())

    def bestbid_volume(self):
        """Returns the available depth at the current best bid"""
        vol = 0
        for order in self.buy_side[self.bestbid()]:
            vol += order[1]
        return vol

    def bestask_volume(self):
        """Returns the available depth at the current best ask"""
        vol = 0
        for order in self.sell_side[self.bestask()]:
            vol += order[1]
        return vol

    def volume_at_price(self, price, side='sell'):
        """
        Returns the available depth on the specified side of the book at the specified price
        :param price: price level to check depth
        :param side: side of the book to check

        :return: the available depth
        """

        vol = 0
        if side == 'sell':
            for order in self.sell_side[price]:
                vol += order[1]
        else:
            for order in self.buy_side[price]:
                vol += order[1]
        return vol

    def buyside_volume(self):
        """Returns total buy side volume"""
        vol = 0
        for price_level in self.buy_side:
            for order in self.buy_side[price_level]:
                vol += order[1]
        return vol

    def sellside_volume(self):
        """Returns total sell side volume"""
        vol = 0
        for price_level in self.sell_side:
            for order in self.sell_side[price_level]:
                vol += order[1]
        return vol

    #@jit(target="cpu")
    def excess_supply(self, normalized=False):
        """
        Returns the Order Imbalance (excess supply)
        :param normalized: if true return the normalized imbalance (defaults to false)

        :return: (Normalized) Order Imbalance
        """

        if not normalized:
            return self.sellside_volume() - self.buyside_volume()
        sell_vol = self.sellside_volume()
        buy_vol = self.buyside_volume()
        return (sell_vol - buy_vol)/(sell_vol + buy_vol)

    def print_book(self):
        """Prints the current state of the book"""
        print('******************************************************')
        print('Buy Side:')
        print(self.buy_side)
        print('--------------------------------------------------')
        print('Sell Side:')
        print(self.sell_side)
        print('******************************************************')

    def submit_limitorder(self, trade_sign, price=None, volume=None, time=None):
        """
        Submits a limit order to the order-book

        :param trade_sign: (1 or -1) the direction of the limit order
        :param price: price at which to submit order (must a multiple of tick size)
        :param volume: order volume (must be a multiple of lot size)
        :param time: time at which the order is submitted (time > lob.time)

        :return: The ID assigned to the limit order
        """
        if price is None:
            if trade_sign == 1:
                price = self.bestbid()
            else:
                price = self.bestask()

        self.check_order_validity(price, volume, time)

        if trade_sign == 1:
            try:
                self.buy_side[price].append([self.next_orderid, volume, time])
                self.time = time
            except KeyError:
                self.buy_side[price] = deque()
                self.buy_side[price].append([self.next_orderid, volume, time])
                self.time = time

            self.limit_orders[self.next_orderid] = [trade_sign, price, volume, time, 'active']
            self.next_orderid += 1
        elif trade_sign == -1:
            try:
                self.sell_side[price].append([self.next_orderid, volume, time])
                self.time = time
            except KeyError:
                self.sell_side[price] = deque()
                self.sell_side[price].append([self.next_orderid, volume, time])
                self.time = time

            self.limit_orders[self.next_orderid] = [trade_sign, price, volume, time, 'active']
            self.next_orderid += 1
        else:
            raise ValueError('Invalid Trade Sign')

        if len(self.buy_side) != 0 and len(self.sell_side) != 0:
            self.match_orders()  # run match orders only if there are orders on both sides

        return self.next_orderid - 1

    def check_order_validity(self, price, volume, time):
        """
        Checks that the price and volume are multiples of the tick and lot sizes and
        checks that the time is later than the latest time in the order book

        :param price: order price
        :param volume: order volume
        :param time: order time

        :return: boolean, true if order is valid false otherwise
        """

        # raise appropriate errors
        if price is not None and (not (round(price / self.tick_size, 5).is_integer()) or price <= 0):
            raise ValueError('The price entered is not permissible')
        if not ((volume / self.lot_size).is_integer()) or volume <= 0:
            raise ValueError('The volume entered is not permissible')
        if time <= self.time:
            raise ValueError('The time entered {} must be greater than the current '
                             'time in the LOB {}'.format(time, self.time))


    def cancel_limitorder(self, order_id, time):
        """
        Cancels an active limit order
        :param order_id: ID of order to be cancelled
        :param time: cancellation time (time > lob.time)
        """
        if time > self.time:
            if self.limit_orders[order_id][4] == 'active':
                if self.limit_orders[order_id][0] == 1:
                    side = self.buy_side
                else:
                    side = self.sell_side

                for order in side[self.limit_orders[order_id][1]]:
                    if order[0] == order_id:
                        order_to_remove = order.copy()
                side[self.limit_orders[order_id][1]].remove(order_to_remove)
                # remove price level if the queue is depleted
                if len(side[self.limit_orders[order_id][1]]) == 0:
                    del side[self.limit_orders[order_id][1]]

                self.limit_orders[order_id][4] = 'cancelled'
            else:
                raise AttributeError('This order is not active')
        else:
            raise ValueError('The time entered must be greater than the current time in the LOB')

    def submit_marketorder(self, trade_sign, volume, time, return_price=False):
        """
        Matches Market orders against active orders by price-time priority. If the market order wipes out the book
        its remaining portion *does not* become an active order.

        :param trade_sign : direction of market order
        :param volume : market order volume
        :param time : time at which order is submitted
        :param return_price: if True, the function returns the total execution price of the market order

        :return: total execution price of the market order
        """
        self.check_order_validity(None, volume, time)
        execution_price = 0
        if trade_sign == 1:
            self.last_market_order_direction = 1
            while volume > 0 and len(self.sell_side) > 0:  # runs while the market order has volume left to execute
                price = self.bestask()
                priority_order = self.sell_side[price][0]
                if priority_order[1] > volume:  # check if the limit order has more volume than the market order
                    priority_order[1] -= volume  # update limit order volume (partial fill)
                    self.limit_orders[priority_order[0]][2] = priority_order[1]  # update volume
                    execution_price += volume * price
                    volume = 0  # update remaining volume

                elif priority_order[1] == volume:  # check if the limit order and market order volume is equal
                    execution_price += volume * price
                    volume = 0  # update market order volume
                    self.sell_side[price].popleft()  # remove limit order
                    self.limit_orders[priority_order[0]][4] = 'executed'  # update status
                    self.limit_orders[priority_order[0]][2] = 0  # update volume

                else:  # case when market order volume is greater than limit order volume
                    volume -= priority_order[1]
                    execution_price += priority_order[1] * price
                    self.sell_side[price].popleft()  # remove limit  order
                    self.limit_orders[priority_order[0]][4] = 'executed'  # update status
                    self.limit_orders[priority_order[0]][2] = 0  # update volume

                # if volume at this price level is depleted, remove price level
                if len(self.sell_side[price]) == 0:
                    del self.sell_side[price]
            self.time = time

            if return_price:
                return execution_price

        elif trade_sign == -1:
            self.last_market_order_direction = -1
            while volume > 0 and len(self.buy_side) > 0:  # runs while the market order has volume left to execute
                price = self.bestbid()
                priority_order = self.buy_side[price][0]
                if priority_order[1] > volume:  # check if the limit order has more volume than the market order
                    priority_order[1] -= volume  # update limit order volume (partial fill)
                    self.limit_orders[priority_order[0]][2] = priority_order[1]  # update volume
                    execution_price += volume * price
                    volume = 0  # update remaining volume

                elif priority_order[1] == volume:  # check if the limit order and market order volume is equal
                    execution_price += volume * price
                    volume = 0  # update market order volume
                    self.buy_side[price].popleft()  # remove limit order
                    self.limit_orders[priority_order[0]][4] = 'executed'  # update status
                    self.limit_orders[priority_order[0]][2] = 0  # update volume

                else:  # case when market order volume is greater than limit order volume
                    volume -= priority_order[1]
                    execution_price += priority_order[1] * price
                    self.limit_orders[priority_order[0]][4] = 'executed'  # update status
                    self.limit_orders[priority_order[0]][2] = 0  # update volume
                    self.buy_side[price].popleft()  # remove limit

                # if volume at this price level is depleted, remove price level
                if len(self.buy_side[price]) == 0:
                    del self.buy_side[price]
            self.time = time

            if return_price:
                return execution_price
        else:
            raise ValueError('Invalid Trade Sign')

    def match_orders(self):
        """
        Order matching procedure to be run on limit order submission. Matches orders that have crossed spread
        """
        while self.bestask() <= self.bestbid():
            priority_ask = self.sell_side[self.bestask()][0]
            priority_bid = self.buy_side[self.bestbid()][0]
            if priority_ask[1] > priority_bid[1]:  # check if sell limit order volume > buy limit order volume
                priority_ask[1] -= priority_bid[1]  # update best ask volume
                self.buy_side[self.bestbid()].popleft()  # remove exhausted limit order

                self.limit_orders[priority_ask[0]][2] = priority_ask[1]  # update volume

                self.limit_orders[priority_bid[0]][4] = 'executed'  # update status
                self.limit_orders[priority_bid[0]][2] = 0  # update volume

            elif priority_ask[1] == priority_bid[1]:
                # remove exhausted limit orders
                self.buy_side[self.bestbid()].popleft()
                self.sell_side[self.bestask()].popleft()

                self.limit_orders[priority_ask[0]][4] = 'executed'  # update status
                self.limit_orders[priority_bid[0]][4] = 'executed'  # update status

                self.limit_orders[priority_ask[0]][2] = 0  # update volume
                self.limit_orders[priority_bid[0]][2] = 0  # update volume

            else:  # case where sell limit order volume < buy limit order volume
                priority_bid[1] -= priority_ask[1]  # update best bid volume
                self.sell_side[self.bestask()].popleft()

                self.limit_orders[priority_ask[0]][4] = 'executed'  # update status
                self.limit_orders[priority_ask[0]][2] = 0  # update volume

                self.limit_orders[priority_bid[0]][2] = priority_bid[1]  # volume

            # if volume at this price level is depleted, remove price level
            if len(self.buy_side[self.bestbid()]) == 0:
                del self.buy_side[self.bestbid()]

            if len(self.sell_side[self.bestask()]) == 0:
                del self.sell_side[self.bestask()]

            # If

    def last_order_sign(self, type):
        """
        returns the trade sign of the last market order or limit order submitted

        :param type: ['limit', 'market'] the type of order for which the sign is retrieved

        :return: +1 or -1, the last trade sign for the specified order type
        """
        if type == 'limit':
            if len(self.limit_orders) == 0:
                return 0
            return self.limit_orders[self.next_orderid - 1][0]
        elif type == 'market':
            return self.last_market_order_direction
        else:
            raise ValueError('type must be on of "limit" or "market"')

    def is_active(self, order_ids):
        """checks if the given order id's are active, returns boolean list"""
        return [self.limit_orders[id][4] == 'active' for id in order_ids]

    def plot(self, width=1):
        """
        Plots the current state of the LOB

        :param width: bar width, to be passed to pyplot.plotbar
        """
        colors = ['b'] * len(self.buy_side) + ['r'] * len(self.sell_side)
        price_levels = list(self.buy_side) + list(self.sell_side)

        buy_vols = []
        sell_vols = []

        for pl in self.buy_side:
            vol = 0
            for order in self.buy_side[pl]:
                vol += order[1]
            buy_vols.append(vol)

        for pl in self.sell_side:
            vol = 0
            for order in self.sell_side[pl]:
                vol += order[1]
            sell_vols.append(-vol)

        depths = buy_vols + sell_vols

        bound = max(max(depths), -min(depths))
        pyplot.bar(price_levels, depths, color=colors, width=width)
        pyplot.ylim([-bound, bound])
        pyplot.show()