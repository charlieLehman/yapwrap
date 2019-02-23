# Copyright Charlie Lehman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Visualizer(object):
    def __init__(self):
        pass

    def add_plot(self):
        raise NotImplementedError

    def get_image(self):
        self.canvas.draw()
        w, h = self.figure.get_size_inches() * self.figure.get_dpi()
        img = np.fromstring(self.canvas.tostring_rgb(),  dtype='uint8').reshape(int(h), int(w), 3)
        return img

class LinePlot(Visualizer):
    def __init__(self, **kwargs):
        super(LinePlot, self).__init__()
        self.kwargs = kwargs
        # These are the "Tableau 20" colors as RGB
        self.colors = np.array([[31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
                       [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
                       [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
                       [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
                       [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]], dtype=np.float32)
        # Normalize colors for matplotlib
        self.colors /= np.array([255.0, 255.0, 255.0])
        self._plot_count = 0
        self.figure, self.axs = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Line Attributes
        self.linewidth = kwargs.get('linewidth', 2.0)
        self.linestyle = kwargs.get('linestyle', '-')

        # Figure Attributes
        self.title = kwargs.get('title', None)
        self.title_fontweight = kwargs.get('title_fontweight', 'bold')
        self.xlabel = kwargs.get('xlabel', None)
        self.ylabel = kwargs.get('ylabel', None)
        self.axisbelow = kwargs.get('axisbelow', True)
        self.legend = kwargs.get('legend', True)
        self.legend_pos = kwargs.get('legend_pos', 1)
        self.grid = kwargs.get('show_grid', True)
        self.grid_style = kwargs.get('grid_style', '--')
        self.grid_width = kwargs.get('grid_width', self.linewidth/2)
        self.labels = []
        self.lines = []


        if self.axisbelow:
            self.axs.set_axisbelow(True)

        self.axs.tick_params(which='both', # Options for both major and minor ticks
                right=False,  # turn off right ticks
                bottom=False) # turn off bottom ticks

        if self.title:
            self.axs.set_title(self.title, fontweight=self.title_fontweight)

        if self.xlabel:
            self.axs.set_xlabel(self.xlabel)
        if self.ylabel:
            self.axs.set_ylabel(self.ylabel)

    def add_plot(self, data, color=None, labels=None, linestyle=None, linewidth=None):
        if linestyle is None:
            linestyle = self.linestyle

        if linewidth is None:
            linewidth = self.linewidth

        if color is None:
            color = self.colors[self._plot_count%self.colors.shape[0], :]

        self._plot_count += 1
        if isinstance(data, np.ndarray):
            assert data.shape[0] == 2
            line = self.axs.plot(data[0, :], data[1, :],
                          color=color,
                          linewidth=linewidth,
                          linestyle=linestyle,
                          label=labels)
            if self.legend:
                self.axs.legend(loc=self.legend_pos)
            if self.grid:
                self.axs.grid(linestyle=self.grid_style, linewidth=self.grid_width)
            self.labels.append(labels)
            self.lines.append(line)

        else:
            raise Exception('data passed to add_plot must be numpy ndarray with dimension 2xN')

class HistPlot(Visualizer):
    def __init__(self, **kwargs):
        super(HistPlot, self).__init__()
        self.kwargs = kwargs
        # These are the "Tableau 20" colors as RGB
        self.colors = np.array([[31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
                       [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
                       [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
                       [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
                       [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229]], dtype=np.float32)
        # Normalize colors for matplotlib
        self.colors /= np.array([255.0, 255.0, 255.0])
        self._plot_count = 0
        self.figure, self.axs = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Figure Attributes
        self.linewidth = kwargs.get('linewidth', 2.0)
        self.title = kwargs.get('title', None)
        self.title_fontweight = kwargs.get('title_fontweight', 'bold')
        self.xlabel = kwargs.get('xlabel', None)
        self.ylabel = kwargs.get('ylabel', None)
        self.axisbelow = kwargs.get('axisbelow', True)
        self.legend = kwargs.get('legend', True)
        self.legend_pos = kwargs.get('legend_pos', 1)
        self.grid = kwargs.get('show_grid', True)
        self.grid_style = kwargs.get('grid_style', '--')
        self.grid_width = kwargs.get('grid_width', 0.5)
        self.labels = []
        self.lines = []

        # Histogram Attributes
        self.histtype = kwargs.get('histtype', 'bar')
        self.align = kwargs.get('align', 'mid')
        self.orientation = kwargs.get('orientation', 'vertical')
        self.rwidth = kwargs.get('rwidth', None)
        self.bins = kwargs.get('bins', 'auto')
        self.range = kwargs.get('range', None)


        if self.axisbelow:
            self.axs.set_axisbelow(True)

        self.axs.tick_params(which='both', # Options for both major and minor ticks
                right=False,  # turn off right ticks
                bottom=False) # turn off bottom ticks

        if self.title:
            self.axs.set_title(self.title, fontweight=self.title_fontweight)

        if self.xlabel:
            self.axs.set_xlabel(self.xlabel)
        if self.ylabel:
            self.axs.set_ylabel(self.ylabel)

    def add_plot(self, data, color=None, labels=None, rwidth=None, orientation=None, histtype=None, bins=None, align=None, range=None):
        if rwidth is None:
            rwidth = self.rwidth
        if orientation is None:
            orientation = self.orientation
        if histtype is None:
            histtype = self.histtype
        if range is None:
            range = self.range
        if bins is None:
            bins = self.bins
        if align is None:
            align = self.align
        if color is None:
            color = color = self.colors[self._plot_count%self.colors.shape[0], :]

        self._plot_count += 1
        if isinstance(data, np.ndarray):
            line = self.axs.hist(data,
                                 color=color,
                                 rwidth=rwidth,
                                 orientation=orientation,
                                 histtype=histtype,
                                 range=range,
                                 bins=bins,
                                 align=align,
                                 label=labels)
            if self.legend:
                self.axs.legend(loc=self.legend_pos)
            if self.grid:
                self.axs.grid(linestyle=self.grid_style, linewidth=self.grid_width)
            self.labels.append(labels)
            self.lines.append(line)

        else:
            raise Exception('data passed to add_plot must be numpy ndarray with dimension 1xN')
