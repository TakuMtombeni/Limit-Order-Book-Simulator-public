# Title: Empirical Properties In The Simulated Limit Order Book
# Author: Taku Mtombeni
#
# The purpose of the script file is to examine the empirical properties of the simulated LOB

# %% Preamble
from src.lobtools import OrderBook
from src.lobtools import Simulator
import pandas as pd
import numpy as np
from models import Model1, Model2, Model3, Model4, Model5
import matplotlib.pyplot as plt
from matplotlib import style
import time as timer
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm, gamma

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

params1a = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 2.378780, 'delta01': 0.1616471,
            'delta10': 1.2053617, 'delta11': 0.2952203, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
            'beta': 0.5895936657995583, 'pi': 0.7955}

params1b = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
            'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
            'beta': 0.5895936657995583, 'pi': 0.7955}

params2a = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 2.378780, 'alpha1': 0.10,
            'alpha2': 0.00002, 'delta10': 1.2053617, 'delta11': 0.2952203, 'rho': 0.6907, 'u': 90, 'vmin': 10,
            'beta': 7, 'pi': 0.8}

params2b = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.55124881, 'alpha1': 0.09138655,
            'alpha2': 1.6203711, 'delta10': 3.11425633, 'delta11': 0.0902544, 'rho': 0.6907, 'u': 90, 'vmin': 10,
            'beta': 7, 'pi': 0.8}

params3 = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
           'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
           'beta': 10.183688838665676, 'pi': 0.7955}

params4 = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
            'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
            'beta': 0.18, 'pi': 0.7955, 'alpha': 79}

params5 = {'decay': 0.01, 'adjacency': adjacency, 'baseline': baseline, 'delta00': 3.12284148, 'delta01': 0.16178544,
           'delta10': 2.56444489, 'delta11': 0.29539388, 'rho': 0.6907, 'u': 89.9719, 'vmin': 10,
           'beta': 8.05708084, 'pi': 0.7955, 'alpha': 23.94028007}

model1 = Model1(parameters=params1b)
model4 = Model4(parameters=params4)

# %% Examine Runtime scaling
simulator = Simulator(lob_init=lob, model=model1)
times = np.arange(60, 10*60*60, 30*60)
run_times = np.zeros(len(times))
num = 10
for t in range(len(times)):
    start = timer.time()
    res_model1 = simulator.run(n=num, obs_freq=0.5, run_time=times[t])
    run_times[t] = (timer.time() - start)/num
    print(t)

#%% run time plot
beta = sum(run_times*times/60)/sum(np.power(times/60, 2))
plt.scatter(times/60, run_times)
plt.plot([0, times[t]/60], [0, beta*times[t]/60], label='y=0.0769x')
plt.ylabel('Mean Run Time (seconds)')
plt.xlabel('Simulated Market Time (minutes)')

plt.legend()
#plt.savefig('Simulation_time.pdf', dpi=2000, bbox_inches='tight')
#plt.close()
plt.show()

 # %% Run Simulations
simulator = Simulator(lob_init=lob, model=model1)
start = timer.time()
res_model1 = simulator.run(n=5, obs_freq=0.1, run_time=7*60*60)
elapsed1 = timer.time() - start

# %% Plot Mid Price and Excess Supply
res_model1[0].excess_supply.plot()
plt.show()

#%% Mid-Price Combined Plot

def mid_price_comb_plot(output):
    for m in range(len(output)):
        output[m].mid_price.resample('30s').pad().plot()
        plt.ylabel('Price (cents)')
        plt.xlabel('Time')
    #plt.savefig('SimulatedPaths.pdf', dpi=2000, bbox_inches='tight')
    #plt.close()
    plt.show()


mid_price_comb_plot(res_model1)

#%% Order Flow Plots
# Spread


def spread_plots(res_model):
    for res in res_model:
        plt.plot(np.arange(0, 7*60*60, 0.1), res.spread.rolling(1000).mean())#.rolling(1000).mean()

    plt.xlabel('Time (Seconds)')
    plt.ylabel('Spread (ticks)')
    #plt.savefig('spread_100ms_samp_model1_multi_100sec_rolling_mean.pdf', dpi=1200, bbox_inches='tight')
    #plt.close()
    plt.show()

    res_model[0].spread.resample('100ms').pad().plot()
    plt.xlabel('Time')
    plt.ylabel('Spread (ticks)')
    #plt.savefig('spread_100ms_samp_model1sec.pdf', dpi=1200, bbox_inches='tight')
    #plt.close()
    plt.show()


spread_plots(res_model1)

#%%
# Volume at Best
res_model1[0].depth_at_best_buy.hist(density=True, bins=15)
plt.show()


#%% Returns


def returns_diagnostics(tau, output, save=False):
    rt = np.log(output.resample(tau).pad().mid_price).diff().dropna()
    plot_acf(rt)
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    if save:
        plt.savefig(tau+'returns_autocorelation.pdf', dpi=1200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    plot_acf(np.abs(rt))
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    if save:
        plt.savefig(tau+'absolute_returns_autocorelation.pdf', dpi=1200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    plt.hist(rt, density=True, bins=15)
    gridxx = np.linspace(min(rt),max(rt),1200)
    plt.plot(gridxx, norm.pdf(gridxx, *norm.fit(rt)))
    if save:
        plt.savefig(tau+'returns_density.pdf', dpi=1200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    qqplot(rt, line='s')
    if save:
        plt.savefig(tau+'returns_qqplot.pdf', dpi=1200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


returns_diagnostics('5s', res_model4[0])
returns_diagnostics('15s', res_model4[0])
returns_diagnostics('30s', res_model4[0])
returns_diagnostics('60s', res_model4[0])
