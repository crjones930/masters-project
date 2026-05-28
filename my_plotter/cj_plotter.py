# my personal module that works to emulate matplotlib for plotting 2D slices for MUSIC

version = "1.0"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib.patches import Arc
from matplotlib.patches import Wedge


class plot2D():
    def __init__(self, t, r, values, vmin=None, vmax=None, visible_ax = True, limit_labels=True, figsize=(18,12), **kwargs):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for direc in self.ax.spines:
            self.ax.spines[direc].set_visible(False)

        r_edge = self.centres_to_edges(r)
        theta_edge = self.centres_to_edges(t)

        R, Theta = np.meshgrid(r_edge, theta_edge, indexing='ij')
        
        x,y = self.polar2imaginarycart(R, Theta)

        if vmin == None:
            vmin = np.min(values)
        if vmax == None:
            vmax = np.min(values)
        
        norm = mplc.Normalize(vmin, vmax)
        self.mesh = self.ax.pcolormesh(x, y, values, norm=norm, rasterized=True, **kwargs)

        self.tlim = np.min(theta_edge), np.max(theta_edge)
        self.rlim = np.min(r_edge), np.max(r_edge)

        if visible_ax == True:
            self.add_arc(self.rlim[0])
            self.add_arc(self.rlim[1])
            self.ax.plot([self.rlim[0] * np.sin(self.tlim[0]),self.rlim[1] * np.sin(self.tlim[0])], [self.rlim[0] * np.cos(self.tlim[0]),self.rlim[1] * np.cos(self.tlim[0])], c='k',lw=1)
            self.ax.plot([self.rlim[0] * np.sin(self.tlim[1]),self.rlim[1] * np.sin(self.tlim[1])], [self.rlim[0] * np.cos(self.tlim[1]),self.rlim[1] * np.cos(self.tlim[1])], c='k',lw=1)

        if limit_labels == True:
            self.label(self.rlim[0], np.format_float_positional(self.rlim[0],precision=4, unique=False, trim='.'))
            self.label(self.rlim[1], np.format_float_positional(self.rlim[1],precision=4, unique=False, trim='.'))
        
        self.fig.canvas.draw()  # optionally update display
    def add_arc(self,r,linestyle='-', label=None, label_position='top', **kwargs):
        self.ax.add_patch(Arc((0,0),2*r,2*r, angle=-90, 
                     theta1=np.degrees(self.tlim[0]), theta2=np.degrees(self.tlim[1]),
                     lw=1, linestyle=linestyle
                     , **kwargs))
        if label != None:
            self.label(r, label, label_position)
    def colorbar(self, **kwargs):
        self.ax.figure.colorbar(self.mesh, ax=self.ax, **kwargs)
    def scatter(self, t, r, s=1, label=None,label_position='top', **kwargs):
        x,y = self.polar2imaginarycart(r,t)
        self.ax.scatter(x, y, s=s,**kwargs)
        if label != None:
            self.label(np.mean(r), label, label_position)
    def plot(self, t, r, lw=1, label=None, label_position='top', **kwargs):
        x,y = self.polar2imaginarycart(r,t)
        self.ax.plot(x, y, lw=1, **kwargs)
        if label != None:
            self.label(np.mean(r), label, label_position)
    def step(self, t, r, lw=1, label=None, label_position='top', **kwargs):
        t_temp = [self.tlim[0]]
        r_temp = [r[0]]
        for i in range(len(t) - 1):
            t_temp.append(t[i])
            r_temp.append(r[i])
            t_temp.append(np.mean([t[i], t[i+1]]))
            r_temp.append(r[i])
            t_temp.append(np.mean([t[i], t[i+1]]))
            r_temp.append(r[i+1])
        t_temp.append(t[-1])
        r_temp.append(r[-1])
        t_temp.append(self.tlim[1])
        r_temp.append(r[-1])
        t_temp = np.array(t_temp)
        r_temp = np.array(r_temp)
        x,y = self.polar2imaginarycart(r_temp,t_temp)
        self.ax.plot(x,y, lw=1, markevery=(1,3), **kwargs)
        if label != None:
            self.label(np.mean(r), label, label_position)
    def label(self, r, text, label_position='top'):
        if label_position == 'top':
            self.ax.text(r * np.sin(self.tlim[0]), r * np.cos(self.tlim[0]), text, ha='right')
        elif label_position == 'bottom':
            self.ax.text(r * np.sin(self.tlim[1]), r * np.cos(self.tlim[1]), text, ha='right', va='top')
    def centres_to_edges(self, centres: np.ndarray) -> np.ndarray:
        """Compute approximate cell edges from the `centres`"""
        dx = np.diff(centres)
        return np.concatenate(
            (centres[:-1] - 0.5 * dx, centres[-2:] + 0.5 * dx[-2:])
        )
    def zone(self, r_in, r_out, **kwargs):
        self.ax.add_patch(Wedge((0,0), r_out, theta1=90 - np.degrees(self.tlim[1]), theta2=90 - np.degrees(self.tlim[0]), width=r_out - r_in, **kwargs))
    def polar2imaginarycart(self, r,t):
        X = r * np.sin(t)
        Y = r * np.cos(t)
        return X, Y
    def _repr_png_(self):
        """Return PNG bytes for Jupyter auto-display."""
        from io import BytesIO
        buf = BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches = 'tight')
        return buf.getvalue()
    def savefig(self, *args,**kwargs):
        self.fig.savefig(*args, **kwargs)