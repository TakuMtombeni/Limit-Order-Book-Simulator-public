from copy import deepcopy
import numpy as np
import pandas as pd


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
                    order = self.model.gen_limitorder(lob)
                    id_list.append(lob.submit_limitorder(*order, event[1]))

                elif event[0] == 'loc':  # limit order cancellation
                    order_to_cancel = self.model.gen_cancelation(lob, id_list)
                    if order_to_cancel >= 0:
                        lob.cancel_limitorder(order_to_cancel, event[1])
                    else:
                        continue  # if nothing happened dont collect data
                elif event[0] == 'moa':  # market order case
                    order = self.model.gen_market_order(lob)
                    lob.submit_marketorder(*order, event[1])

                else:
                    count += 1
                    mid_price[count] = lob.mid_price()
                    vol_best_bid[count] = lob.bestbid_volume()
                    vol_best_ask[count] = lob.bestask_volume()
                    order_imbalance[count] = lob.excess_supply()
                    spread[count] = lob.spread()

            outputs[k] = pd.DataFrame(index=pd.to_timedelta(time, unit='s'),
                                      data={'mid_price': mid_price, 'spread': spread, 'excess_supply': order_imbalance,
                                            'depth_at_best_buy': vol_best_bid, 'depth_at_best_sell':vol_best_ask})
        return outputs


