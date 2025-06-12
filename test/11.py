import pandas as pd
from matplotlib import pyplot as plt


a = pd.read_pickle('C:\\PycharmProjects\\PythonProject\\Gen_SMFS\\test\\clustering_xp_14_sim_speed_500_data.csv')
b = pd.read_pickle('C:\\PycharmProjects\\PythonProject\\Gen_SMFS\\test\\clustering_Fu_14_sim_speed_500_data.csv')
c = pd.read_pickle('C:\\PycharmProjects\\PythonProject\\Gen_SMFS\\test\\clustering_WLC_data_14_sim_speed_500_data.csv')

print(len(a))
print(c['Force'].max())

print(c.keys())
for i in range(10):
    data = a.iloc[i]
    data = data[:-4]
    print(type(data.values))
    plt.plot(data, b.iloc[i][:-4])
plt.show()

a = pd.read_pickle('clustering_Fu_1_sim_speed_500_data.csv')
a = a[:-4]

plt.plot(a.iloc[1])
plt.show()

