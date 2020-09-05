# Time Independent Spread with Linear Liquidity Taking + intercept term

import numpy as np
from tick.hawkes import SimuHawkesExpKernels
import math


def get_active_orders(lob):
    active_orders = []
    for price_level in lob.buy_side:
        for order in lob.buy_side[price_level]:
            active_orders.append(order[0])
    for price_level in lob.sell_side:
        for order in lob.sell_side[price_level]:
            active_orders.append(order[0])
    return active_orders


class Model4:

    def __init__(self, parameters):
        """

        :param parameters: a dictionary of parameters required by the model
        """
        self.parameters = parameters

    def gen_event_times(self, run_time):
        decays = self.parameters['decay']
        adjacency = self.parameters['adjacency']
        baseline = self.parameters['baseline']

        hawkes_sim = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False)
        hawkes_sim.end_time = run_time
        hawkes_sim.simulate()

        # Generate Event Arrival Times
        limit_order_arrivals = hawkes_sim.timestamps[1]
        limit_order_arrivals = [('loa', time) for time in limit_order_arrivals]

        limit_order_cancellations = hawkes_sim.timestamps[2]
        limit_order_cancellations = [('loc', time) for time in limit_order_cancellations]

        market_order_arrivals = hawkes_sim.timestamps[0]
        market_order_arrivals = [('moa', time) for time in market_order_arrivals]

        # Merge into single array of event times
        event_times = limit_order_arrivals.copy()
        event_times.extend(limit_order_cancellations)
        event_times.extend(market_order_arrivals)
        event_times.sort(key=lambda x: x[1])

        return event_times

    def gen_market_order(self, lob):
        beta = self.parameters['beta']
        alpha = self.parameters['alpha']
        pi = self.parameters['pi']

        last_trade_sign = lob.last_market_order_direction
        if last_trade_sign == 0:
            last_trade_sign = np.random.choice([1, -1])
        trade_sign = last_trade_sign * np.random.choice([1, -1], p=[pi, 1 - pi])

        if trade_sign == 1:  # make Pv dependent on best_ask_volume
            mu = alpha + beta * lob.bestask_volume()
        else:  # make Pv dependent on best_bid_volume
            mu = alpha + beta * lob.bestbid_volume()

        volume = (np.random.poisson(mu) + 1) * lob.lot_size

        return trade_sign, volume

    def gen_limit_order(self, lob):
        delta00 = self.parameters['delta00']
        delta01 = self.parameters['delta01']
        delta10 = self.parameters['delta10']
        delta11 = self.parameters['delta11']
        rho = self.parameters['rho']
        u = self.parameters['u']

        # Get required LOB State Data
        buy_side_vol = lob.buyside_volume()
        sell_side_vol = lob.sellside_volume()
        bestbid = lob.bestbid()
        bestask = lob.bestask()
        tau = lob.tick_size

        # Generate Direction
        eta = rho * (sell_side_vol - buy_side_vol) / (sell_side_vol + buy_side_vol)
        p = 1 / (1 + math.exp(-eta))
        direction = np.random.binomial(1, p)
        if direction == 0:
            direction = - 1

        # Generate Price
        log_spread = math.log((bestask - bestbid) / tau)
        s1 = np.exp(delta00 + delta01 * log_spread)
        s2 = np.exp(delta10 + delta11 * log_spread)
        delta = np.random.poisson(s1) - np.random.poisson(s2)

        if direction == 1:
            price = bestbid - delta * tau
        else:
            price = bestask + delta * tau

        # Generate Volume
        Pv = 1 / u
        volume = np.random.geometric(Pv) * lob.lot_size

        return direction, round(price, 2), volume

    def gen_cancellation(self, lob, id_list):
        vmin = self.parameters['vmin']
        num_sells = sum([len(value) for key, value in lob.sell_side.items()])
        num_buys = sum([len(value) for key, value in lob.buy_side.items()])
        # cancel random order if condition met
        if num_sells > vmin and num_buys > vmin:
            return np.random.choice(get_active_orders(lob))
        else:
            return -1
