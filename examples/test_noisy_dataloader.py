from yapwrap.dataloaders import NoisyDataloader, CIFAR10
from matplotlib import pyplot as plt
import numpy as np


cifar10 = CIFAR10()
noisy_cifar10 = NoisyDataloader(CIFAR10(), p=1.00)

x = cifar10.examples
y = noisy_cifar10.examples

x = np.transpose(x,(0,2,3,1))
y = np.transpose(y,(0,2,3,1))

fig, ax = plt.subplots(2)
ax[0].imshow(x[0])
ax[1].imshow(y[0])
plt.show()

