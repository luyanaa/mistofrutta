from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class IndexTracker(object):
    def __init__(self, ax, X, colors=['blue','cyan','green','orange','red','magenta'], cmap='', Overlay=[]):
        self.ax = ax
        ax.set_title('Scroll to navigate in z\nA-D to change channel, W-S to change color scaling')
        
        new_keys_set = {'a', 'd', 'w', 's'}
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
        
        #Build colormaps
        self.Cmap = []
        if cmap=='':
            self.ncolors = len(colors)
            for q in np.arange(self.ncolors):
                cmap = LinearSegmentedColormap.from_list('name', ['black',colors[q]])
                self.Cmap.append(cmap)
        else:
            self.ncolors = 1
            self.Cmap.append(cmap)
        
        #Extract informations about the stack
        self.X = X
        self.dimensions = X.shape
        self.slices = self.dimensions[-3]
        rows = self.dimensions[-2]
        cols = self.dimensions[-1]
        if len(self.dimensions) == 4: 
            self.channels = self.dimensions[-4]
        
        self.OverlayData = Overlay
        
        #Initialize z position and channel for first plot
        self.z = self.slices//2
        self.ch = 0
        self.scale = 1.0
        
        #Plot and initialize vmax
        if len(self.dimensions) == 4:
            self.im = ax.imshow(self.X[self.ch, self.z, ...]/self.scale,cmap=self.Cmap[self.ch],interpolation='none')
        else:
            self.im = ax.imshow(self.X[self.z, ...]/self.scale,cmap=self.Cmap[self.ch], interpolation='none')
        
        try:
            if self.ch != 4: 
                goodcolor='r'
            else:
                goodcolor='g'
            punti = self.OverlayData[self.z].T
            self.overlay, = ax.plot(punti[0],punti[1],'o',markersize=1,c=goodcolor)
        except:
            print("no overlay")
        
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.z = (self.z + 1) % self.slices
        else:
            self.z = (self.z - 1) % self.slices
        self.update()
        
    def onkeypress(self, event):
        if event.key == 'd':
            self.ch = (self.ch + 1) % self.channels
        elif event.key == 'a':
            self.ch = (self.ch - 1) % self.channels
        elif event.key == 's':
            self.scale = min(self.scale + 0.1, 1.0)
        elif event.key == 'w':
            self.scale = max(self.scale - 0.1, 0.0001)
        self.update()

    def update(self):
        if len(self.dimensions) == 4: 
            self.im.set_data(self.X[self.ch ,self.z, ...]/self.scale)
        else:
            self.im.set_data(self.X[self.z, ...]/self.scale)
        self.ax.set_xlabel('slice %s' % self.z)
        self.im.set_cmap(self.Cmap[self.ch])
        
        try:
            if self.ch != 4: 
                goodcolor='r'
            else:
                goodcolor='g'
            punti = self.OverlayData[self.z].T
            self.overlay.set_xdata(punti[0])
            self.overlay.set_ydata(punti[1])
        except:
            pass        
        
        self.im.axes.figure.canvas.draw()
        #self.im.axes.figure.canvas.draw()


def hyperstack(A,cmap='',Overlay=[]):
    '''
    Parameters
    ----------
    A: numpy array
        Hyperstack [channel, z, x, y] or [z, x, y]
    
    Returns
    -------
    None
    
    Takes a hyperstack with shape [channel, z, x, y] or [z, x, y] and
    produces the GUI to navigate it.    
    '''
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, A, cmap=cmap, Overlay=Overlay)
    fig.canvas.mpl_connect('key_press_event', tracker.onkeypress)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    fig.clear()
    del tracker
 
#import unmix as um
#filename = "/home/francesco/tmp/unmixing/SU_180503/SU_180503_AKS153.3.iii_W2Z"
#Image, Brightfield = um.load_image(filename) # Indices are [z,channel,x,y]
#Image = um.sort_image_leica(Image)
#Image = np.swapaxes(Image, 0,1)
#plot_hyperstack(Image)
    
