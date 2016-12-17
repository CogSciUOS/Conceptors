import numpy as np
import numpy.random
import matplotlib.pyplot as plt

a = np.random.random((16, 16))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()