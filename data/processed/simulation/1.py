import matplotlib.pyplot as plt
import numpy as np

a = np.load('fe_curves.npy')

for i in range(5):
    plt.plot(a[i])

plt.show()