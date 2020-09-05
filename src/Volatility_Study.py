#%% Preamble
from LOBTools import OrderBook
import numpy as np
from scipy.stats import skellam, gamma, norm, kurtosis
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from matplotlib import style, cm, rc
from tick.hawkes import SimuHawkesExpKernels
import time as timer
from pygam import GammaGAM, s ,te

rc('text', usetex=True)
style.use('seaborn-darkgrid')
#%% Simulation Helper Functions
def gen_market_order(lob, beta=0.75, pi=0.8):
    
    
    last_trade_sign = lob.last_market_order_direction 
    if last_trade_sign == 0:
        last_trade_sign = np.random.choice([1, -1])
    trade_sign = last_trade_sign * np.random.choice([1,-1], p=[pi, 1-pi])


    if trade_sign == 1: # make Pv dependent on best_ask_volume
        mu = beta*lob.bestask_volume()
    else: # make Pv dependent on best_bid_volume
        mu = beta*lob.bestbid_volume()

    volume = (np.random.poisson(mu)+1)*lob.lot_size

    return (trade_sign, volume)

def gen_limitorder(lob, delta00=2.378780, delta01=0.1616471, delta10=1.2053617, delta11=0.2952203, rho=0.001, u = 90):
    

    eta = rho*lob.excess_supply()/(lob.buyside_volume()+lob.sellside_volume())
    p = np.exp(eta)/(1+np.exp(eta))
    direction = np.random.binomial(1, p)
    if direction == 0:
        direction = - 1


    spread = lob.spread()/lob.tick_size
    s1 = np.exp(delta00 + delta01*np.log(spread))
    s2 = np.exp(delta10 + delta11*np.log(spread))
    delta = np.random.poisson(s1) - np.random.poisson(s2)

    Pv = 1/u
    volume = np.random.geometric(Pv)*lob.lot_size

    if direction == 1:
        price = lob.bestbid() - delta*lob.tick_size
    else:
        price = lob.bestask() + delta*lob.tick_size

    return (direction, round(price,2), volume)

def get_active_orders(lob):
    active_orders = []
    for price_level in lob.buy_side:
        for order in lob.buy_side[price_level]:
            active_orders.append(order[0])
    for price_level in lob.sell_side:
        for order in lob.sell_side[price_level]:
            active_orders.append(order[0])
    return active_orders
#%%
def simulate(parameters, T, adjacency, decay):
    '''
    parameters = (pi, beta, rho, u, delta00, delta01, delta10, delta11, mu1, mu2, mu3)
    '''

    pi, beta, rho, u, delta00, delta01, delta10, delta11, mu1, mu2, mu3 = parameters
    baseline = np.array([mu1, mu2, mu3])

    # Simulate Times
    hawkes_sim = SimuHawkesExpKernels(adjacency=adjacency,decays=decay, baseline=baseline, verbose=False)
    hawkes_sim.end_time = T
    hawkes_sim.simulate()

    # Generate Event Arrival Times
    limit_order_arivals = hawkes_sim.timestamps[1]
    limit_order_arivals = [('loa', time) for time in limit_order_arivals]

    limit_order_cancelations = hawkes_sim.timestamps[2]
    limit_order_cancelations = [('loc', time) for time in limit_order_cancelations]

    market_order_arivals = hawkes_sim.timestamps[0]
    market_order_arivals = [('moa', time) for time in market_order_arivals]

    # Merge into single array of event times
    event_times = limit_order_arivals.copy()
    event_times.extend(limit_order_cancelations) 
    event_times.extend(market_order_arivals)
    event_times.sort(key = lambda x: x[1])

    #%% initialize order book
    lob = OrderBook('XYZ', tick_size=1)

    AAPL_LOB = pd.read_csv( # Orderbook
    '../../Data/LOBSTER_SampleFile_AAPL_2012-06-21_50/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv',
     header=None
    )

    init = AAPL_LOB.iloc[0]
    t = 10**(-10)
    dt = 10**(-10)
    #Initial Orders
    for i in range(0,AAPL_LOB.shape[1],4):
        if init[i+1] > 0:
            lob.submit_limitorder(-1,init[i]/100,init[i+1],t)
            t += dt
    for i in range(2, AAPL_LOB.shape[1],4):
        if init[i+1] > 0:
            lob.submit_limitorder(1,init[i]/100,init[i+1],t)
            t += dt

    
    midprice = [lob.mid_price()]
    spread = [lob.spread()]
    time = [lob.time]

    for event in event_times:
        # simulation
        if event[0] == 'loa':  # limit order arrival
            order = gen_limitorder(lob,  delta00, delta01, delta10, delta11, rho, u)
            ID = lob.submit_limitorder(*order, event[1])
        elif event[0] == 'loc':  # limit order cancellation
            # get number of sell and buy orders 
            num_sells = sum([len(value) for key, value in lob.sell_side.items()])
            num_buys = sum([len(value) for key, value in lob.buy_side.items()])
            # cancel random order if condition met
            if(num_sells > 5 and num_buys > 5):
                active_orders = get_active_orders(lob)
                lob.cancel_limitorder(np.random.choice(active_orders), event[1])
            else:
                continue # if nothing happended dont collect data
    
        elif event[0] == 'moa':  # market order case
            order = gen_market_order(lob, beta, pi)
            lob.submit_marketorder(*order, event[1])
        else:
            raise ValueError('Invalid event type')

        midprice.append(lob.mid_price())
        spread.append(lob.spread())
        time.append(lob.time)

    Output = pd.DataFrame(index=pd.to_timedelta(time, unit='s'), 
                        data={'mid_price':midprice, 'spread':spread})
    
    sigma5 = np.std(np.log(Output.resample('5s').first().mid_price).diff().dropna())
    sigma30 = np.std(np.log(Output.resample('30s').first().mid_price).diff().dropna())
    sigma60 = np.std(np.log(Output.resample('60s').first().mid_price).diff().dropna())
    meanSpread = np.mean(Output.spread)

    return (sigma5, sigma30, sigma60, meanSpread, *parameters)

#%%
T = 60*60  #200*60
N = 1000

#params = (pi, beta, rho, u, delta00, delta01, delta10, delta11, mu1, mu2, mu3)
#params = (0.8, 0.75, 0.6907, 90, 2.378780, 0.1616471, 1.2053617, 0.2952203, 0.56498618, 1.13601817, 1.04177811)
col_names = ['sigma5', 'sigma30', 'sigma60', 'meanSpread', 'pi', 
'beta', 'rho', 'u', 'delta00', 'delta01', 'delta10', 'delta11', 'mu1', 'mu2', 'mu3']

decays = 0.01

adjacency = np.array([[0.58846155, 0.        , 0.        ],
                      [1.53428232, 0.14604008, 0.66436009],
                      [1.41833894, 0.08804415, 0.6706122 ]])

res_matrix = np.zeros((N, len(col_names)))
for i in range(N):
    lb = [0,    0, 0,   1,   2, 0.1,  0.8,  0.2, 0.2, 1, 0.2]
    ub = [1, 0.95, 4, 120, 3.1, 0.3, 1.55, 0.38, 0.8, 2, 1.2]
    params = np.random.uniform(lb, ub)
    res_matrix[i] = simulate(params, T, adjacency, decays)
    print(i)

res = pd.DataFrame(res_matrix, columns=col_names)


# %%
features = ['pi', 'beta', 'rho', 'u', 'delta00', 'delta01', 'delta10', 'delta11', 'mu1', 'mu2', 'mu3']
texnamesx = ['$\\pi$', '$\\beta$', '$\\rho$', '$u$', '$\\delta_{00}$', '$\\delta_{01}$', '$\\delta_{10}$', '$\\delta_{11}$', '$\\mu_1$', '$\\mu_2$', '$\\mu_3$']
texnamesy = ['$f_1(\\pi)$', '$f_2(\\beta)$', '$f_3(\\rho)$', '$f_4(u)$', '$f_5(\\delta_{00})$', '$f_6(\\delta_{01})$', '$f_7(\\delta_{10})$', '$f_8(\\delta_{11})$', '$f_9(\\mu_1)$', '$f_10(\\mu_2)$', '$f_11(\\mu_3)$']

fit5 = GammaGAM().fit(res[features], res.sigma5)
fit30 = GammaGAM(terms=s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7)+s(8)+s(9)+s(10)+te(9,10)+te(1,3)).fit(res[features], res.sigma30)
fit30 = fit30.gridsearch(res[features], res.sigma30, lam=np.logspace(3,4,100))
fit60 = GammaGAM().fit(res[features], res.sigma60)
fitSpread = GammaGAM().fit(res[features], res.meanSpread)
# %% Volatility 30 Plots
x_grid = np.array([np.linspace(min(res[feature]), max(res[feature])*1, 
1000) for feature in features])

for k in range(len(features)):
    pdeps, cof = fit30.partial_dependence(k, width=0.95, X = x_grid.T)

    fig, ax = plt.subplots()
    ax.plot(x_grid[k], pdeps, '-')
    ax.fill_between(x_grid[k], cof[:,0], cof[:,1], alpha=0.2)
    plt.xlabel(texnamesx[k])
    plt.ylabel(texnamesy[k])
    plt.savefig(features[k] +'_partial_dependence_plot.pdf', dpi=1200, bbox_inches = 'tight')
    plt.close()

Z = fit30.partial_dependence(11, meshgrid=True)
X, Y = fit30.generate_X_grid(11, meshgrid=True)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.ylabel('$\mu_3$')
plt.xlabel('$\mu_2$')
plt.savefig('mu1_mu2_partial_dependence_plot.pdf', dpi=1200,bbox_inches = 'tight')
plt.close()

Z = fit30.partial_dependence(12, meshgrid=True)
X, Y = fit30.generate_X_grid(12, meshgrid=True)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.ylabel('$u$')
plt.xlabel('$\\beta$')
plt.savefig('u_beta_partial_dependence_plot.pdf', dpi=1200, bbox_inches = 'tight')
plt.close()

# %% Diagnostics Volatility
resids30 = fit30.deviance_residuals(res[features],res.sigma30)
yhat30 = fit30.predict(res[features])

plt.scatter(yhat30,resids30, s=5)
plt.ylabel('Residuals')
plt.xlabel('Predicted')
plt.savefig('diagnostics_residuals_v_predicted.pdf', dpi=1200, bbox_inches = 'tight')
plt.close()

plt.scatter(yhat30, res.sigma30, s=5)
plt.plot([0,0.00025], [0,0.00025], 'r--')
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.savefig('diagnostics_observed_v_predicted.pdf', dpi=1200, bbox_inches = 'tight')
plt.close()

plot_acf(resids30)
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.savefig('diagnostics_residual_acf.pdf', dpi=1200, bbox_inches = 'tight')
plt.close()

qqplot(resids30, fit=True, line='45')
plt.savefig('diagnostics_residual_qqplot.pdf', dpi=1200, bbox_inches = 'tight')
plt.close()

#%% Diagnositics Spread
residsSpread = fitSpread.deviance_residuals(res[features],res.meanSpread)
yhatSpread = fitSpread.predict(res[features])

plt.scatter(residsSpread, yhatSpread)
plt.xlabel('Residuals')
plt.ylabel('Predicted')
plt.show()

plt.scatter(yhatSpread, res.meanSpread)
plt.plot([0,10000], [0,10000], 'r-')
plt.show()
# %%
# %% Spread Plots
x_grid = np.array([np.linspace(min(res[feature]), max(res[feature])*1, 
1000) for feature in features])

for k in range(len(features)):
    pdeps, cof = fitSpread.partial_dependence(k, width=0.95, X = x_grid.T)

    fig, ax = plt.subplots()
    ax.plot(x_grid[k], pdeps, '-')
    ax.fill_between(x_grid[k], cof[:,0], cof[:,1], alpha=0.2)
    plt.xlabel(features[k])

    plt.show()


# %%
