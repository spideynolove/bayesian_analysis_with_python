import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data\\mauna_loa_CO2.csv', delimiter=',')
plt.plot(data[:, 0], data[:, 1])
plt.xlabel('$year$', fontsize=16)
plt.ylabel('$CO_2 (ppmv)$', fontsize=16)
plt.show()