from yapwrap.utils import visualize
import numpy as np
from scipy.misc import imshow

# Random Plot data
points = np.linspace(0, 1000, 100000, endpoint=True)
d2 = np.cos(0.01*points)
d3 = np.sin(0.03*points)
d1 = (d3+d2)/2
# d3 = np.linspace(0, t, int(fs*t), endpoint=False)
x = np.vstack((points, d1))
y = np.vstack((points+20, d2))
z = np.vstack((points+50, d3))


# # LinePlot API
lp = visualize.LinePlot(title='My Plot',
                        xlabel='X Label',
                        ylabel='Y Label',
                        legend=True,
                        legend_pos=1,
                        grid=True)
lp.add_plot(x, label='Data1', linewidth=3)
lp.add_plot(y, label='Data2')
lp.add_plot(z, label='Data3', linestyle='--', linewidth=1)
lp.update_plot(-z, 'Data3', linewidth=5)
img1 = lp.get_image()
lp.close()

# Show result
# imshow(img)

d1 = np.random.laplace(loc=15, scale=3, size=50)
d2 = np.random.laplace(loc=10, scale=3, size=50)
d3 = np.random.laplace(loc=20, scale=3, size=50)
hp = visualize.HistPlot(title='My Plot',
                        xlabel='X Label',
                        ylabel='Y Label',
                        legend=True,
                        legend_pos=1,
                        grid=True)
hp.add_plot(d1, label='D1')
hp.add_plot(d2, label='D2')
hp.add_plot(d3, label='D3')
hp.add_plot(d1-6, label='D4')
hp.add_plot(d2-5, label='D5')
hp.add_plot(d3-15, label='D6')
hp.add_plot(d1-11, label='D7')
hp.add_plot(d2+11, label='D8')
hp.add_plot(d3+3, label='D9')
hp.add_plot(d1+2, label='D10')
hp.add_plot(d2+4, label='D11')
hp.add_plot(d3+10, label='D12')
hp.add_plot(np.random.normal(loc=10, scale=5, size=25), label='D13')
hp.add_plot(np.random.normal(loc=11, scale=5, size=25), label='D14')
hp.add_plot(np.random.normal(loc=12, scale=5, size=25), label='D15')
hp.add_plot(np.random.normal(loc=13, scale=5, size=25), label='D16')
hp.add_plot(np.random.normal(loc=14, scale=5, size=25), label='D17')
hp.add_plot(np.random.normal(loc=15, scale=5, size=25), label='D18')
hp.add_plot(np.random.normal(loc=16, scale=5, size=25), label='D19')
hp.add_plot(np.random.normal(loc=17, scale=5, size=25), label='D20')
hp.update_plot(d3+0.25, label='D12')
img2 = hp.get_image()
hp.close()

# Show result
imshow(img1)
imshow(img2)
