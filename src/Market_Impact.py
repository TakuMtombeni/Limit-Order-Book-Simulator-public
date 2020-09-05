# Title: Optimal Placement In The Simulated Limit Order Book
# Author: Taku Mtombeni
#
# The purpose of the script file is to test market impact in the simulated market

# %% Preamble
from src.lobtools import OrderBook
from src.lobtools import Simulator
from models import Model1, Model2
import matplotlib.pyplot as plt
from matplotlib import style
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
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
# Initial Orders
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
#0.5895936657995583,

model1 = Model1(parameters=params1)

#%% Time Exploration 1
simulator = Simulator(lob_init=lob, model=model1)
start = time.time()
test_buy100 = simulator.impact_test(1000, (1, 100, 0.00001), 61*60, 10, True)
test_buy50 = simulator.impact_test(1000, (1, 50, 0.00001), 61*60, 10, True)
test_sell50 = simulator.impact_test(1000, (-1, 50, 0.00001), 61*60, 10, True)
test_sell100 = simulator.impact_test(1000, (-1, 100, 0.00001), 61*60, 10, True)
elapsed = time.time() - start

#%% plot time exploration results
fig, ax = plt.subplots()
test_buy100.plot(ax=ax)
test_buy50.plot(ax=ax)
test_sell50.plot(ax=ax)
test_sell100.plot(ax=ax)
ax.legend(['Buy, Size = 100', 'Buy, Size = 50', 'Sell, Size = 50', 'Sell, Size = 100'])
plt.xlabel('time')
plt.ylabel('Mean Impact (cents)')
#plt.savefig('impact_60min_model1_time_10sec_samp_n500.pdf', dpi=1200, bbox_inches='tight')
#plt.close()
plt.show()

#%% Time exploration 2

test_buy1 = simulator.impact_test(5000, (1, 1, 0.00001), 15*60+1, 5, True)
test_buy1.plot()
plt.show()


#%% Size Exploration
simulator = Simulator(lob_init=lob, model=model1)

Sizes = np.arange(50, 700, 50)
res = [0]*len(Sizes)
for i in range(len(Sizes)):
    res[i] = simulator.impact_test(10000, (1, Sizes[i], 0.00001), 60+2, 30)
    print(i)

#%% Plots
t = 2  # time in minutes
y = np.array([np.mean(result, axis=0)[t] for result in res])
plt.scatter(Sizes, y)
plt.scatter([0], [0])
plt.xlabel('Order Size')
plt.ylabel('Mean Impact (cents)')
plt.show()
#plt.savefig('impact_1min_model1_sizes_sell_morder_n20000.pdf', dpi=1200, bbox_inches='tight')
#plt.close()

#%% Curve Fit

beta = sum(Sizes*y)/sum(np.power(Sizes, 2))
plt.scatter(Sizes, [np.mean(result, axis=0)[t] for result in res])
plt.plot([0, 700], [0, 700*beta], label='y = 0.038x')
plt.scatter([0], [0])
plt.xlabel('Order Size')
plt.ylabel('Mean Impact (cents)')
plt.legend()
plt.savefig('impact_1min_model1_high_large_sizes_buy_morder_n10000.pdf', dpi=1200, bbox_inches='tight')
plt.close()
#plt.show()

#%% Size Exploration short
simulator = Simulator(lob_init=lob, model=model1)

Sizes = np.arange(1, 51, 1)
res_short = [0]*len(Sizes)
for i in range(len(Sizes)):
    res_short[i] = simulator.impact_test(50000, (1, Sizes[i], 0.00001), 65, 5)
    print(i)

#%% Plots
t = 1  # time in seconds
plt.scatter(Sizes, [np.mean(result, axis=0)[t] for result in res_short])
plt.scatter([0], [0])
plt.xlabel('Order Size')
plt.ylabel('Mean Impact (cents)')
#plt.savefig('impact_size_5sec_model1_10000.pdf', dpi=1200, bbox_inches='tight')
#plt.close()
plt.show()

#%%
regres = sm.OLS([np.mean(result, axis=0)[t] for result in res_short], sm.add_constant(Sizes)).fit()
regres.params

#%% DELETE THIS
#%% Size Exploration
simulator = Simulator(lob_init=lob, model=model1)

Sizes = np.arange(50, 700, 50)
res = [0]*len(Sizes)
for i in range(len(Sizes)):
    res[i] = simulator.impact_test(2000, (1, None, Sizes[i], 0.00001), 10*60+2, 60)
    print(i)

#%% Plots
t = 2  # time in minutes
y = np.array([np.mean(result, axis=0)[t] for result in res])
plt.scatter(Sizes, y)
plt.scatter([0], [0])
plt.xlabel('Order Size')
plt.ylabel('Mean Impact (cents)')
plt.show()
#plt.savefig('impact_1min_model1_sizes_sell_morder_n20000.pdf', dpi=1200, bbox_inches='tight')
#plt.close()

#%% Curve Fit

beta = sum(Sizes*y)/sum(np.power(Sizes, 2))
plt.scatter(Sizes, [np.mean(result, axis=0)[t] for result in res])
plt.plot([0, 700], [0, 700*beta], label='y = 0.002x')
plt.scatter([0], [0])
plt.xlabel('Order Size')
plt.ylabel('Mean Impact (cents)')
plt.legend()
plt.savefig('impact_10min_model1_high_large_sizes_buy_lorder_n2000.pdf', dpi=1200, bbox_inches='tight')
plt.close()
plt.show()