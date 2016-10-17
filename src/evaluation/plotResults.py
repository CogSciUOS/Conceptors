
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

x = [4, 2, 1, 0.5]
y = [5, 6, 7]
z = [72.051535564244332, 76.514586115293099, 65.118681057682338, 83.603219052657721, 84.223189418032163, 85.91552705756753, 86.135222227292545, 83.197517160841585, 73.7431687578094, 85.622110646275772, 75.373368151354867, 73.042649241343184]

fig = figure()
ax = axes(projection='3d')

#z = np.linspace(0, 1, 100)
#x = z * np.sin(20 * z)
#y = z * np.cos(20 * z)

c = x + y

ax.scatter(x, y, z, c=c)

show()