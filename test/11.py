import pandas as pd
from matplotlib import pyplot as plt

a = pd.read_pickle('D:\\PYTHON\\project\\Gen_SMFS\\test\\clustering_xp_14_sim_speed_500_data.csv')
b = pd.read_pickle('D:\\PYTHON\\project\\Gen_SMFS\\test\\clustering_Fu_14_sim_speed_500_data.csv')
c = pd.read_pickle('D:\\PYTHON\\project\\Gen_SMFS\\test\\clustering_WLC_data_14_sim_speed_500_data.csv')

print(len(a))
print(c['Force'].max())

print(c.keys())
for i in range(10):
    data = a.iloc[i]
    data = data[:-4]
    print(type(data.values))
    plt.plot(data, b.iloc[i][:-4])
plt.show()
