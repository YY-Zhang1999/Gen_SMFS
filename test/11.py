import pandas as pd
from matplotlib import pyplot as plt

a = pd.read_pickle('clustering_Fu_1_sim_speed_500_data.csv')


plt.plot(a[10])
plt.show()