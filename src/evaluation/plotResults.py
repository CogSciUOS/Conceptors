from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import *

noiseRange = [4,2,1,0.5]
numSyllRange = np.arange(5, 15, 2).tolist()


noise_p = []
num_syll_p = []

for noise in noiseRange:
    for numSyll in numSyllRange:
        noise_p.append(noise)
        num_syll_p.append(numSyll)

print(noise_p)
print(num_syll_p)
perf = [79.093473245462363,
        74.101934494351696,
        64.660285102741341,
        69.731564916956231,
        56.068248720087958,
        78.207545113484485,
        67.465657980537173,
        75.62773417198639,
        61.629249890705395,
        69.774712137296888,
        83.028525274032162,
        74.071023978632951,
        76.633968061562683,
        59.136179130010078,
        63.909365692661297,
        67.75312733026378,
        75.059045212555844,
        74.669930373688516,
        65.513667976155631,
        61.346718103413743]


fig = figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(noise_p, num_syll_p, perf, c='r', marker='o')
ax.set_xlabel('Noise')
ax.set_ylabel('# of Syllables')
ax.set_zlabel('Performance')
show()