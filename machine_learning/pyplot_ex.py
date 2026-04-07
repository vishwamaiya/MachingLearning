# Demonstrating the use of matplotlib.pyplot to plot a simple line graph of GDP growth over time.
import matplotlib.pyplot as plt

x = [2001, 2006, 2011, 2016, 2021, 2026]
gdp_val_growth = [1.7, 1.9, 2.1, 3.7, 3.59, 5.0]

plt.plot(x, gdp_val_growth, marker='o')

plt.show()
