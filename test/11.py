import pandas as pd
from matplotlib import pyplot as plt

a = pd.read_pickle('clustering_Fu_1_sim_speed_500_data.csv')
a = a[:-4]

plt.plot(a.iloc[1])
plt.show()