from yapwrap.utils import visualize
import numpy as np
from scipy.misc import imshow

# Random Plot data
points = np.linspace(0, 100, 10000, endpoint=True)
d2 = np.cos(0.01*points)
d3 = np.sin(0.03*points)
d1 = (d3+d2)/2
# d3 = np.linspace(0, t, int(fs*t), endpoint=False)
x = np.vstack((points, d1))
y = np.vstack((points, d2))
z = np.vstack((points, d3))


# LinePlot API
lp = visualize.LinePlot(title='My Plot',
                        xlabel='X Label',
                        ylabel='Y Label',
                        legend=True,
                        legend_pos=1,
                        grid=True)
lp.add_plot(x, labels='Data1')
lp.add_plot(y, labels='Data2')
lp.add_plot(z, labels='Data3', linestyle='--', linewidth=1)
img1 = lp.get_image()

# Show result
# imshow(img)

d1 = np.random.laplace(loc=15, scale=3, size=500)
d2 = np.random.laplace(loc=10, scale=3, size=500)
d3 = np.random.laplace(loc=20, scale=3, size=500)
hp = visualize.HistPlot(title='My Plot',
                        xlabel='X Label',
                        ylabel='Y Label',
                        legend=True,
                        legend_pos=1,
                        grid=True)
hp.add_plot(d1, labels='D1', rwidth=0.8)
hp.add_plot(d2, labels='D2', rwidth=0.8)
hp.add_plot(d3, labels='D2', rwidth=0.8)
img2 = hp.get_image()

# Show result
imshow(img1)
imshow(img2)
