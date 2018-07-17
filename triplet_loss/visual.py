import numpy as np
import visdom


class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        super(VisdomLinePlotter, self).__init__()
        self.viz = visdom.Visdom()
        assert self.viz.check_connection()
        self.env = env_name
        self.plots = {}

    def plot(self, win, name, x, y):
        if win not in self.plots:
            self.plots[win] = self.viz.line(
                X = np.array([x]),
                Y = np.array([y]),
                env = self.env,
                name = name,
                opts = dict(
                    xlabel = 'epochs',
                    ylabel = win,
                    xtickmin = 1,
                    showlegend = True
                )
            )
        else:
            self.viz.line(
                X = np.array([x]),
                Y = np.array([y]),
                env = self.env,
                win = self.plots[win],
                name = name,
                update = 'append'
            )


if __name__ == '__main__':
    vis = VisdomLinePlotter('visual')
    import time
    for i in range(10):
        vis.plot('accuracy', 'epoch', i, i)
        time.sleep(1)
