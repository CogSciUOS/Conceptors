from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import *

noiseRange = [4,2,1,0.5]
numSyllRange = np.arange(4,8,1).tolist()

noise_p = []
num_syll_p = []

for noise in noiseRange:
    for numSyll in numSyllRange:
        noise_p.append(noise)
        num_syll_p.append(numSyll)

print(noise_p)
print(num_syll_p)
perf = [84.55901186907613,
        85.567966897884986,
        83.989491483760119,
        74.938845312680456,
        76.619100152833568,
        76.946928257038508,
        90.04792988879484,
        80.51562354563174,
        74.824476590452193,
        83.524931739165552,
        80.397902338927693,
        85.139805162615758,
        87.285524021102674,
        78.574101204208503,
        80.009318638840185,
        84.091205358299788]

fig = figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(noise_p, num_syll_p, perf, c='r', marker='o')
ax.set_xlabel('Noise')
ax.set_ylabel('# of Syllables')
ax.set_zlabel('Performance')
show()