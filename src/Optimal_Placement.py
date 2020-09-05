# Title: Optimal Placement In The Simulated Limit Order Book
# Author: Taku Mtombeni
#
# The purpose of the script file is ...
# %% Preamble
from statsmodels.distributions.empirical_distribution import ECDF
from src.lobtools import OrderBook
from src.lobtools import Simulator
from models import Model1, Model4, Model5
import matplotlib.pyplot as plt
from matplotlib import style
from copy import deepcopy
import pandas as pd
import numpy as np
import progressbar
import pickle
import time

style.use('seaborn-darkgrid')

# %% Initialize Simulation
lob = OrderBook('XYZ', tick_size=1, lot_size=1)

AAPL_LOB = pd.read_csv(
 './data/LOBSTER_SampleFile_AAPL_2012-06-21_50/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv',
 header=None
)

init = AAPL_LOB.iloc[5000]
t = 10**(-10)
dt = 10**(-10)

# Load Initial Orders
for i in range(0, AAPL_LOB.shape[1], 4):
    if init[i+1] > 0:
        lob.submit_limitorder(-1, init[i]/100, init[i+1], t)
        t += dt
for i in range(2, AAPL_LOB.shape[1], 4):
    if init[i+1] > 0:
        lob.submit_limitorder(1, init[i]/100, init[i+1], t)
        t += dt
print(lob.ticker)

adjacency = np.array([[0.58846155, 0.        , 0.        ],
                      [1.53428232, 0.14604008, 0.66436009],
                      [1.41833894, 0.08804415, 0.6706122 ]])

baseline = np.array([0.56498618, 1.13601817, 1.04177811])


params1 = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
           'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
           'beta': 0.5895936657995583, 'pi': 0.7955}

params1low = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
              'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
              'beta': 0.05, 'pi': 0.7955}

params1high = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
               'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
               'beta': 0.90, 'pi': 0.7955}

params2b = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
            'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
            'beta': 0, 'pi': 0.7955, 'alpha': 25}

params5 = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
           'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
           'beta': 8.05708084, 'pi': 0.7955, 'alpha': 23.94028007}

model1_normal = Model1(parameters=params1)
model1_high = Model1(parameters=params1low)
model1_low = Model1(parameters=params1high)
model2 = Model4(parameters=params2b)
model5 = Model5(parameters=params5)
#%% Gen Out flows function

def gen_outflows(n, model, t, dt, lob_init):
    """
    Generates sample of n outflows for the given model during [t, t+dt]

    :param n: number of outflows to generate
    :param model: model object to use for simulation
    :param t: beginning of time interval to measure outflows
    :param dt: end of time interval to measure outflows
    :return: np.array of observed outflows
    """
    outflows = np.zeros(n)
    bar = progressbar.ProgressBar(max_value=n - 1)
    for k in range(n):
        lob = deepcopy(lob_init)
        event_times = model.gen_event_times(t + dt)
        event_times.append(('Spec', t))  # append Beginning of Time interval
        event_times.sort(key=lambda x: x[1])
        id_list = []

        for event in event_times:
            # simulation
            if event[0] == 'loa':  # limit order arrival
                order = model.gen_limit_order(lob)
                id_list.append(lob.submit_limitorder(*order, event[1]))
            elif event[0] == 'loc':  # limit order cancellation
                order_to_cancel = model.gen_cancellation(lob, id_list)
                if order_to_cancel >= 0:
                    lob.cancel_limitorder(order_to_cancel, event[1])
                else:
                    continue  # if nothing happened dont collect data
            elif event[0] == 'moa':  # market order case
                order = model.gen_market_order(lob)
                lob.submit_marketorder(*order, event[1])
            else:
                b0 = lob.bestbid()
                ID0 = [order[0] for order in lob.buy_side[b0]]
                V0 = np.array([lob.limit_orders[order_id][2] for order_id in ID0])

        V1 = np.zeros(len(ID0))
        for j in range(len(ID0)):
            if lob.limit_orders[ID0[j]][4] != 'cancelled':
                V1[j] = lob.limit_orders[ID0[j]][2]

        outflows[k] = sum(V0 - V1)
        bar.update(k)
    return outflows

# %% Define Strategies
market_order_bm = lambda lob, S: (S, 0)
limit_order_bm = lambda lob, S: (0, S)


def cont_kukanov_solution(c, f, r, theta, ecdf, X):
    if c != 0:
        def dec_func_cont_kukanov(lob, S):
            Q = lob.bestbid_volume()
            h = 0.5*lob.spread()
            lam_u = h + f + c
            if h + f > lam_u:
                print('condition failed')
            #print('h=%.4f'%(h))
            lam_bot = (2*h + f + r)/ecdf(Q+S) - (h + r + theta)
            #print(lam_bot)
            lam_top = (2*h + f + r)/ecdf(Q) - (h + r + theta)
            if lam_u <= lam_bot:
                return 0, S
            elif lam_u >= lam_top:
                return S, 0
            else:
                #print('here Q=%i'%(Q))
                M = np.round(S - np.quantile(X, (2*h + f + r)/(lam_u + h + r + theta)) + Q)
                L = S - min(M, S)
                return min(M, S), L
    else:
        def dec_func_cont_kukanov(lob, S):
            Q = lob.bestbid_volume()
            h = 0.5*lob.spread()
            lam_min = h + f + np.finfo(float).eps
            #print('h=%.4f'%(h))
            lam_bot = (2*h + f + r)/ecdf(Q+S) - (h + r + theta)
            #print(lam_bot)
            lam_top = (2*h + f + r)/ecdf(Q) - (h + r + theta)
            if lam_min <= lam_bot:
                return 0, S
            elif lam_min >= lam_top:
                return S, 0
            else:
                #print('here Q=%i'%(Q))
                M = np.round(S - np.quantile(X, (2*h + f + r)/(lam_min + h + r + theta)) + Q)
                L = S - min(M, S)
                return min(M, S), L
    return dec_func_cont_kukanov


# %% Initialize simulations
t = 0.0000001
dt = 60
size = 500
fee = 0.29
rebate = 0.24
theta_low = 0.033
theta_normal = 0.034
theta_high = 0.038
n_place = 50000
n_outflow = 10000
outflow_lob = deepcopy(lob)

# pad depth at best
for padn in range(0, 4):
    outflow_lob.submit_limitorder(1, lob.bestbid(), 100, outflow_lob.time + np.finfo(float).eps)



# %% ECDF gen

Xis60_low = gen_outflows(n_outflow, model1_low, t, dt, outflow_lob)
Xis60_normal = gen_outflows(n_outflow, model1_normal, t, dt, outflow_lob)
Xis60_high = gen_outflows(n_outflow, model1_high, t, dt, outflow_lob)

ecdf60_low = ECDF(Xis60_low)
ecdf60_normal = ECDF(Xis60_normal)
ecdf60_high = ECDF(Xis60_high)


# %% ECDF and EPDF plot
xgrid = np.arange(0, 1000)

plt.plot(xgrid, ecdf60_low(xgrid), label='beta = 0.05')
plt.plot(xgrid, ecdf60_normal(xgrid), label='beta = 0.59')
plt.plot(xgrid, ecdf60_high(xgrid), label='beta = 0.90')
plt.xlabel('Outflow')
plt.ylabel('ECDF')
plt.legend()
#plt.savefig('ECDFs_dt60_n10000_q1000.pdf', dpi=2000, bbox_inches='tight')
#plt.close()
plt.show()


plt.hist(Xis60_low, density=True, bins=10, alpha=0.5, label='beta = 0.05')
plt.hist(Xis60_normal, density=True, bins=10, alpha=0.5, label='beta = 0.59')
plt.hist(Xis60_high,  density=True, bins=10, alpha=0.5, label='beta = 0.90')
plt.legend()
#plt.savefig('PDFs_dt60_n10000_q1000.pdf', dpi=2000, bbox_inches='tight')
#plt.close()
plt.show()

#%% Gen Out flows
#Xis = pd.read_csv('./data/Outflows_14m_to_15m_def_params_model1.csv')['0'].to_numpy()

func_low = cont_kukanov_solution(0, fee, rebate, theta_low, ecdf60_low, Xis60_low)
func_normal = cont_kukanov_solution(0, fee, rebate, theta_normal, ecdf60_normal, Xis60_normal)
func_high = cont_kukanov_solution(0, fee, rebate, theta_high, ecdf60_high, Xis60_high)

strats_in_low = {'market_order_bench': market_order_bm, 'limit_order_bench': limit_order_bm, 'cont_kukanov': func_low}
strats_in_normal = {'market_order_bench': market_order_bm, 'limit_order_bench': limit_order_bm, 'cont_kukanov': func_normal}
strats_in_high = {'market_order_bench': market_order_bm, 'limit_order_bench': limit_order_bm, 'cont_kukanov': func_high}

simulator_low = Simulator(lob_init=lob, model=model1_low)
simulator_normal = Simulator(lob_init=lob, model=model1_normal)
simulator_high = Simulator(lob_init=lob, model=model1_high)
#%% Test Placement
#t = 60
res_low = simulator_low.placement_test(strats_in_low, t, dt, size, fee, rebate, n_place)
res_normal = simulator_normal.placement_test(strats_in_normal, t, dt, size, fee, rebate, n_place)
res_high = simulator_high.placement_test(strats_in_high, t, dt, size, fee, rebate, n_place)


#%% Results
def res_print(res, name):
    print('-'*35)
    print('Mean Execution Costs ' + name)
    print('-'*35)
    print(res.mean(axis=0))
    print('-'*35)
    print('Standard Deviations ' + name)
    print('-'*35)
    print(res.std(axis=0))

res_print(res_low, 'beta = 0.05')
res_print(res_normal, 'beta = 0.59')
res_print(res_high, 'beta = 0.90')

#%% over sizes
t = 60
sizes = np.arange(100, 650, 25)
n = 10000
out_s_low = [0]*len(sizes)
out_s_normal = [0]*len(sizes)
out_s_high = [0]*len(sizes)
for i in range(len(sizes)):
    out_s_low[i] = simulator_low.placement_test(strats_in_low, t, dt, sizes[i], fee, rebate, n)
    out_s_normal[i] = simulator_normal.placement_test(strats_in_normal, t, dt, sizes[i], fee, rebate, n)
    out_s_high[i] = simulator_high.placement_test(strats_in_low, t, dt, sizes[i], fee, rebate, n)
    print(i)
# %% over sizes plots
def plotter(out_s, name=None):
    plt.plot(sizes[:i], [out_i.cont_kukanov.mean(axis=0) for out_i in out_s], 'o--b', label='cont-kukanov')
    plt.plot(sizes[:i], [out_i.limit_order_bench.mean(axis=0) for out_i in out_s], 'D--g', label='lim-bench')
    plt.plot(sizes[:i], [out_i.market_order_bench.mean(axis=0) for out_i in out_s], 's--m', label='mark-bench')
    plt.xlabel('Order Size')
    plt.ylabel('Mean Cost Per Share (cents)')
    plt.legend()
    if name:
        plt.savefig(name+'_mean_cost.pdf', dpi=2000, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    plt.plot(sizes[:i], [out_i.cont_kukanov.std(axis=0) for out_i in out_s], 'o--b', label='cont-kukanov')
    plt.plot(sizes[:i], [out_i.limit_order_bench.std(axis=0) for out_i in out_s], 'D--g', label='lim-bench')
    plt.plot(sizes[:i], [out_i.market_order_bench.std(axis=0) for out_i in out_s], 's--m', label='mark-bench')
    plt.xlabel('Order Size')
    plt.ylabel('Standard Deviation')
    plt.legend()
    if name:
        plt.savefig(name+'_std.pdf', dpi=2000, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


plotter(out_s_low[:i], 'low_order_size_dt60_n1000')
plotter(out_s_normal[:i], 'normal_order_size_dt60_n1000')
plotter(out_s_high[:i], 'high_order_size_dt60_n1000')

#%% Save/Load Data

#pickle.dump(out_s, open('placement_res_sizes_50_500_10_n5000_model1_defparam_dt60.p', 'wb'))
#out_s = pickle.load(open('placement_res_sizes_50_500_10_n5000_model1_defparam.p', 'rb'))

