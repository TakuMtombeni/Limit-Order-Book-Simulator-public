# Title: Empirical Properties In The Simulated Limit Order Book
# Author: Taku Mtombeni
#
# The purpose of the script file is to examine the empirical properties of the simulated LOB

# %% Preamble
from src.lobtools import OrderBook
from src.lobtools import Simulator
import pandas as pd
import numpy as np
from models import Model1
import matplotlib.pyplot as plt
from matplotlib import style
import time as timer

style.use('seaborn-darkgrid')

# %% Initialize Simulation
lob = OrderBook('XYZ', tick_size=1, lot_size=1)

AAPL_LOB = pd.read_csv(
 './data/LOBSTER_SampleFile_AAPL_2012-06-21_50/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv',
 header=None
)

init = AAPL_LOB.iloc[0]
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

params = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 2.378780, 'delta01': 0.1616471,
          'delta10': 1.2053617, 'delta11': 0.2952203, 'rho': 0.6907, 'u': 90, 'vmin': 10, 'beta': 0.75, 'pi': 0.8}

model = Model1(parameters=params)
simulator = Simulator(lob_init=lob, model=model)

start = timer.time()
test = simulator.run(n=5, obs_freq=0.5, run_time=1*60*60)
elapsed = timer.time() - start

# %% Plot Mid Price and Excess Supply
test[0].mid_price.plot()
plt.show()

test[0].excess_supply.plot()
plt.show()

#%%
test[0].mid_price.plot()
test[1].mid_price.plot()
test[2].mid_price.plot()
test[3].mid_price.plot()
test[4].mid_price.plot()
plt.show()