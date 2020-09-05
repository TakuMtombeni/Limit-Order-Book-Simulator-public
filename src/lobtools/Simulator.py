from copy import deepcopy
import numpy as np
import pandas as pd
import progressbar


class Simulator:

    def __init__(self, lob_init, model):
        """

        :param lob_init: limit order book initial state
        :param model: model class, specifies model to use to simulate,
        model must posses methods gen_event_times, gen_market_order(lob), gen_limit_order(lob),
        gen_cancel_limit_order(lob)
        """

        self.model = model
        self.lob_init = lob_init

    def run(self, n, obs_freq, run_time):
        """
        Runs Simulation

        :param n: Number of times to run simulation
        :param obs_freq: frequency (in seconds) at which to collect data
        :param run_time: run time for each simulation
        :return: N pandas data frames
        """
        outputs = [0]*n
        for k in range(n):
            # initialize limit order book
            lob = deepcopy(self.lob_init)

            # generate event times
            event_times = self.model.gen_event_times(run_time)
            sampling_times = np.arange(obs_freq, run_time, obs_freq)
            event_times.extend([('samp', time) for time in sampling_times])
            event_times.sort(key=lambda x: x[1])

            # Initialize storage
            mid_price = np.zeros(len(sampling_times)+1)
            vol_best_bid = np.zeros(len(sampling_times) + 1)
            vol_best_ask = np.zeros(len(sampling_times) + 1)
            order_imbalance = np.zeros(len(sampling_times) + 1)
            spread = np.zeros(len(sampling_times) + 1)

            time = [lob.time]
            time.extend(sampling_times)
            mid_price[0] = lob.mid_price()
            vol_best_bid[0] = lob.bestbid_volume()
            vol_best_ask[0] = lob.bestask_volume()
            order_imbalance[0] = lob.excess_supply()
            spread[0] = lob.spread()
            id_list = []

            count = 0
            # step through event times
            for event in event_times:
                # simulation
                if event[0] == 'loa':  # limit order arrival
                    order = self.model.gen_limit_order(lob)
                    id_list.append(lob.submit_limitorder(*order, event[1]))

                elif event[0] == 'loc':  # limit order cancellation
                    order_to_cancel = self.model.gen_cancellation(lob, id_list)
                    if order_to_cancel >= 0:
                        lob.cancel_limitorder(order_to_cancel, event[1])
                    else:
                        continue  # if nothing happened dont collect data
                elif event[0] == 'moa':  # market order case
                    order = self.model.gen_market_order(lob)
                    lob.submit_marketorder(*order, event[1])

                elif event[0] == 'samp':
                    count += 1
                    mid_price[count] = lob.mid_price()
                    vol_best_bid[count] = lob.bestbid_volume()
                    vol_best_ask[count] = lob.bestask_volume()
                    order_imbalance[count] = lob.excess_supply()
                    spread[count] = lob.spread()

                else:
                    raise ValueError("Invalid Order Type, ensure model.gen_event_times is correctly specified")

            outputs[k] = pd.DataFrame(index=pd.to_timedelta(time, unit='s'),
                                      data={'mid_price': mid_price, 'spread': spread, 'excess_supply': order_imbalance,
                                            'depth_at_best_buy': vol_best_bid, 'depth_at_best_sell': vol_best_ask})
        return outputs

    def placement_test(self, strats, t, dt, size, fee, rebate, n):
        """
        Tests a placement strategy on the interval [t, t+dt]

        :param strats: a dict of strategies to be tested,
        each strat is a function that accepts the lob and returns a tupple (M, L), the keys are names of the strats
        :param t: time at which first order is placed
        :param dt: time at which second order is placed
        :param size: order size to be executed
        :param fee: market order fee
        :param rebate: limit order rebate
        :param n: number of simulations
        :return: nxp pandas data frame, where p is the number of strategies
        """
        costs = {}  # dict to store costs
        for strat in strats:
            costs[strat] = np.zeros(n)

        def after_t(dec_func, lob_init, event_times):
            lob = deepcopy(lob_init)
            specID = -2  # will store ID of limit order
            price_lim = 0  # will store price at which limit order was submitted
            Lfilled = 0  # will store number of limit orders shares executed
            Mprice1 = 0  # total execution price of first market order
            Mprice2 = 0  # total execution price of second market order
            remainder = 0  # will store remainder
            dtau = 0.0000000001
            for event in event_times:
                # simulation
                if event[0] == 'loa':  # limit order arrival
                    order = self.model.gen_limit_order(lob)
                    id_list.append(lob.submit_limitorder(*order, event[1]))

                elif event[0] == 'loc':  # limit order cancellation
                    while True:
                        order_to_cancel = self.model.gen_cancellation(lob, id_list)
                        if order_to_cancel != specID:
                            break
                    if order_to_cancel >= 0:
                        lob.cancel_limitorder(order_to_cancel, event[1])
                    else:
                        continue  # if nothing happened dont collect data
                elif event[0] == 'moa':  # market order case
                    order = self.model.gen_market_order(lob)
                    lob.submit_marketorder(*order, event[1])
                elif event[0] == 'Spec0':
                    (M, L) = dec_func(lob, size)
                    #print((M, L))
                    if M > 0:
                        Mprice1 = lob.submit_marketorder(1, M, event[1], True)
                    if L > 0:
                        price_lim = lob.bestbid()
                        specID = lob.submit_limitorder(1, price_lim, L, event[1] + dtau)

                elif event[0] == 'Spec1':
                    if specID != -2:
                        Lfilled = (L - lob.limit_orders[specID][2])
                        #print(Lfilled, lob.limit_orders[specID][4])
                        if M + Lfilled < size:
                            # print(lob.limit_orders[specID])
                            remainder = size - (M + Lfilled)
                            Mprice2 = lob.submit_marketorder(1, remainder, event[1], True)
                            lob.cancel_limitorder(specID, event[1] + dtau)

            return (fee * M + fee * remainder - rebate * Lfilled + price_lim * Lfilled + Mprice1 + Mprice2) / size

        bar = progressbar.ProgressBar(max_value=n - 1)
        for i in range(n):
            lob = deepcopy(self.lob_init)
            event_times = self.model.gen_event_times(t+dt)
            event_times.append(('Spec0', t))
            event_times.append(('Spec1', t + dt))
            event_times.sort(key=lambda x: x[1])
            indx = [time[1] for time in event_times].index(t)
            event_times_before_t = event_times[:indx]
            event_times_after_t = event_times[indx:]
            id_list = []
            for event in event_times_before_t:
                # simulation
                if event[0] == 'loa':  # limit order arrival
                    order = self.model.gen_limit_order(lob)
                    id_list.append(lob.submit_limitorder(*order, event[1]))

                elif event[0] == 'loc':  # limit order cancellation
                    order_to_cancel = self.model.gen_cancellation(lob, id_list)
                    if order_to_cancel >= 0:
                        lob.cancel_limitorder(order_to_cancel, event[1])
                    else:
                        continue  # if nothing happened dont collect data
                elif event[0] == 'moa':  # market order case
                    order = self.model.gen_market_order(lob)
                    lob.submit_marketorder(*order, event[1])

            for strat in strats:
                costs[strat][i] = after_t(strats[strat], lob, event_times_after_t)
            bar.update(i)
        return pd.DataFrame(costs)

    def impact_test(self, n, order_in, run_time, obs_freq, aggregate=False):
        """
        Generates n sample paths of the impact function of an order

        :param n: number of simulations to run
        :param order_in: order tuple (sign, size, time) for market orders, (sign, price, size, time) for limit orders
        :param run_time: time to run the simulation for
        :param obs_freq: frequency at which to observe midprice
        :param aggregate: if true the mean realization is returned, otherwise all n realizations are returned
        :return: realizations (or mean realization) of impact function
        """
        sampling_times = np.arange(obs_freq, run_time, obs_freq)
        impact_sample_paths = [0] * n

        time = [0]
        time.extend(sampling_times)
        bar = progressbar.ProgressBar(max_value=n-1)
        for k in range(n):
            imp = False  # first iteration is case without order x
            midprice_paths = [0, 0]

            event_times = self.model.gen_event_times(run_time)
            event_times.extend([('samp', time) for time in sampling_times])
            for j in range(2):  # second iteration is case without order x
                if j == 1:
                    imp = True

                if imp:
                    if len(order_in) == 3:
                        order_arrival_time = order_in[2]
                    elif len(order_in) == 4:
                        order_arrival_time = order_in[3]
                    else:
                        raise ValueError('Expected Tuple of Length 3 or 4')
                    event_times.append(('spec', order_arrival_time))  # append user specified market order arrival
                event_times.sort(key=lambda x: x[1])

                # initialize order book
                lob = deepcopy(self.lob_init)
                midprice = np.zeros(len(sampling_times) + 1)
                midprice[0] = lob.mid_price()
                count = 1
                id_list = []
                # Run Simulation
                for event in event_times:
                    # simulation
                    if event[0] == 'loa':  # limit order arrival
                        order = self.model.gen_limit_order(lob)
                        id_list.append(lob.submit_limitorder(*order, event[1]))
                    elif event[0] == 'loc':  # limit order cancellation
                        order_to_cancel = self.model.gen_cancellation(lob, id_list)
                        if order_to_cancel >= 0:
                            lob.cancel_limitorder(order_to_cancel, event[1])
                        else:
                            continue  # if nothing happened dont collect data
                    elif event[0] == 'moa':  # market order case
                        order = self.model.gen_market_order(lob)
                        lob.submit_marketorder(*order, event[1])
                    elif event[0] == 'samp':
                        midprice[count] = lob.mid_price()
                        count += 1
                    elif event[0] == 'spec':  # user specified order
                        if len(order_in) == 3:
                            lob.submit_marketorder(*order_in)
                        else:
                            lob.submit_limitorder(*order_in)
                    else:
                        raise ValueError("Invalid Order Type, ensure model.gen_event_times is correctly specified")

                midprice_paths[j] = np.array(midprice)

            impact_sample_paths[k] = midprice_paths[1] - midprice_paths[0]
            bar.update(k)
        if aggregate:
            return pd.DataFrame(index=pd.to_timedelta(time, unit='s'),
                                data={'impact': np.mean(impact_sample_paths, axis=0)})
        return impact_sample_paths
