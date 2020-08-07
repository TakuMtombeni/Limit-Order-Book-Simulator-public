# Title: Empirical Properties In The Simulated Limit Order Book
# Author: Taku Mtombeni
#
# The purpose of the script file is to examine the empirical properties of the simulated LOB

# %% Preamble
from src.lobtools import OrderBook
import pandas as pd
from models import model1
import matplotlib.pyplot as plt

# %% Initialize Simulation
lob = OrderBook('XYZ', tick_size=1, lot_size=1)

AAPL_LOB = pd.read_csv(
 '../../data/LOBSTER_SampleFile_AAPL_2012-06-21_50/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv',
 header=None
)

init = AAPL_LOB.iloc[0]
t = 10**(-10)
dt = 10**(-10)
# Initial Orders
for i in range(0,AAPL_LOB.shape[1],4):
    if init[i+1] > 0:
        lob.submit_limitorder(-1,init[i]/100,init[i+1],t)
        t += dt
for i in range(2, AAPL_LOB.shape[1],4):
    if init[i+1] > 0:
        lob.submit_limitorder(1,init[i]/100,init[i+1],t)
        t += dt
print(lob.ticker)