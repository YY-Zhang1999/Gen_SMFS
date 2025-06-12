import numpy as np
from matplotlib import pyplot as plt

output = np.load('./output/output.npy')

print(output.shape)

for i in range(output.shape[0]):
    plt.plot(output[i].reshape(-1))
    plt.show()



