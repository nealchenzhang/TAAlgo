import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sys


def price_simulation(initial_price, expected_return_cc, sigma, num_days=1000):
    price = np.zeros(num_days)
    price[0] = initial_price

    shock = np.zeros(num_days)
    drift = np.zeros(num_days)

    for x in range(1, num_days):
        shock[x] = sigma * np.random.normal(0, 1)*np.sqrt(1/num_days)
        drift[x] = expected_return_cc * 1/num_days
        price[x] = price[x - 1] + (price[x - 1] * (drift[x] + shock[x]))

    return price

price = price_simulation(2810, 0.03, 0.02)
plt.plot(price)

sim_data_set = pd.DataFrame(columns=list(range(0, 100)))

i = 0
for mu in np.linspace(-0.5, 0.5, 10):
    for sigma in np.linspace(0.2, 0.6, 10):
        sim_data_set.iloc[:, i] = price_simulation(2810, mu, sigma)
        i += 1

sim_data_set.plot()
