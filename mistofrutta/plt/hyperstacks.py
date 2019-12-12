from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from PyQt5.QtCore import pyqtRemoveInputHook

class IndexTracker(object):
    def __init__(self, ax, X, colors=['blue','cyan','green','orange','red','magenta'], cmap='', Overlay=[], OverlayLabels=[]):
        self.selectpointsmode = False
        self.labeledPoints = {}
        pyqtRemoveInputHook()
        
        self.markpointsmode = False
        self.markedPointsN = 0
        
        self.ax = ax
        self.defaultTitle = "Press h for instructions"
        ax.set_title(self.defaultTitle)
        
        self.instructions = '\n**INSTRUCTIONS:\n'+\
            'Scroll/up-down to navigate in z\n'+\
            'A-D/left-right to change channel, W-S to change color scaling \n'+\
            'Ctrl+p for point labeling \n'+\
            'Ctrl+o for point marking (without labels)\n'+\
            'Ctrl+0 minimum of colorscale to 0 (to clip negative values)\n'+\
            'Ctrl+9 minimum of colorscale to original \n'+\
            'Ctrl+1 minimum of colorscale to 100 (for raw images with no binning) \n'+\
            'Ctrl+4 minimum of colorscale to 400 (for raw images with 2x2 binning: whole brain imager, MCW)'
        
        new_keys_set = {'a', 'd', 'w', 's', 'up', 'down','left','right'}
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
        
        #Build colormaps
        self.Cmap = []
        self.colors = colors
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
        self.channels = 1
        if len(self.dimensions) == 4: 
            self.channels = self.dimensions[-4]
        if self.channels > self.ncolors:
            for j in np.arange(self.channels-self.ncolors):
                cmap = LinearSegmentedColormap.from_list('name', ['black','white'])
                self.Cmap.append(cmap)
        
        self.OverlayData = Overlay
        self.OverlayLabels = OverlayLabels
        self.OverlayLabelDx = int(X.shape[-1]/100)
        self.OverlayLabelDy = int(X.shape[-2]/100)
        self.OverlayTxtObj = []
        
        #Initialize z position and channel for first plot
        self.z = self.slices//2
        self.ch = 0
        self.scale = np.ones(self.channels)
        self.maxX = np.max(self.X)
        self.minX = np.min(self.X)
        self.vmax = self.maxX
        
        #Plot and initialize vmax
        if len(self.dimensions) == 4:
            self.im = ax.imshow(self.X[self.ch, self.z, ...],cmap=self.Cmap[self.ch],interpolation='none',vmin=self.minX,vmax=self.vmax) #/self.scale[self.ch]
        else:
            self.im = ax.imshow(self.X[self.z, ...],cmap=self.Cmap[self.ch], interpolation='none',vmin=self.minX,vmax=self.vmax) #/self.scale[self.ch]
        
        try:
            if self.ch != 4: #TODO check self.colors instead of the number of the channel
                goodcolor='r'
            else:
                goodcolor='g'
            if(isinstance(self.OverlayData[0],list)):
                punti = self.OverlayData[self.ch][self.z].T
            else:
                punti = self.OverlayData[self.z].T
                labs = self.OverlayLabels[self.z]
            self.overlay, = ax.plot(punti[0],punti[1],'o',markersize=1,c=goodcolor)
            for i in np.arange(len(labs)):
                ann = self.ax.annotate(labs[i], xy=(punti[0][i],punti[1][i]), xytext=(punti[0][i]+self.OverlayLabelDx,punti[1][i]+self.OverlayLabelDy), color=goodcolor)
                self.OverlayTxtObj.append(ann)
        except:
            pass
        
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.z = (self.z + 1) % self.slices
        else:
            self.z = (self.z - 1) % self.slices
        self.update()
        
    def onkeypress(self, event):
        if not self.selectpointsmode:
            if event.key == 'h':
                print(self.instructions)
            elif event.key == 'd' or event.key == 'right':
                self.ch = (self.ch + 1) % self.channels
            elif event.key == 'a' or event.key == 'left':
                self.ch = (self.ch - 1) % self.channels
            elif event.key == 's':
                self.scale[self.ch] = min(self.scale[self.ch]*1.3, 1.0)
            elif event.key == 'w':
                self.scale[self.ch] = max(self.scale[self.ch]*0.7, 0.0001)
            elif event.key == 'up':
                self.z = (self.z + 1) % self.slices
            elif event.key == 'down':
                self.z = (self.z - 1) % self.slices
            elif event.key == 'ctrl+0':
                self.im.norm.vmin = 0
            elif event.key == 'ctrl+9':
                self.im.norm.vmin = self.minX
            elif event.key == 'ctrl+1':
                self.im.norm.vmin = 100
            elif event.key == 'ctrl+4':
                self.im.norm.vmin = 400
            elif event.key == 'ctrl+o':
                self.markpointsmode = not self.markpointsmode
                if self.markpointsmode == True:
                    if len(self.OverlayData) == 0:
                        for z in np.arange(self.slices):
                            self.OverlayData.append(np.array([[0.0,0.0]]))
                    if self.colors[self.ch] != 'r': 
                        goodcolor='r'
                    else:
                        goodcolor='g'
                    punti = self.OverlayData[self.z].T
                    if not hasattr(self,'overlay'):
                        self.overlay, = self.ax.plot(punti[0],punti[1],'o',markersize=1,c=goodcolor)
                    self.markpointsmodetitle = "Mark points mode. Press ctrl+o to switch back to normal mode.\nRight click to delete last point"
                    self.ax.set_title(self.markpointsmodetitle+"\nMarked points in this plane: "+str(len(self.OverlayData[self.z])-1)+"\nMarked points in the stack: "+str(self.markedPointsN))
                else:
                    self.ax.set_title(self.defaultTitle)
        
        if event.key == 'ctrl+p':
            self.selectpointsmode = not self.selectpointsmode
            if self.selectpointsmode == True:
                self.ax.set_title("Select points mode. Press ctrl+p to switch back to normal mode.")
            else:
                self.ax.set_title(self.defaultTitle)
            
        self.update()
        
    def onbuttonpress(self, event):
        if self.selectpointsmode:
            ix, iy = event.xdata, event.ydata
            if ix != None and iy != None:
                label = input("Enter label for ("+str(int(ix))+","+str(int(iy))+"):  ")
                self.labeledPoints[label] = [self.ch, self.z, int(iy), int(ix)]
        
        elif self.markpointsmode:
            ix, iy = event.xdata, event.ydata
            if ix != None and iy != None and event.button == 1:
                self.OverlayData[self.z] = np.append(self.OverlayData[self.z],np.array([[ix,iy]]),axis=0)
                self.markedPointsN += 1
            elif event.button == 3 and len(self.OverlayData[self.z]) > 1:
                self.markedPointsN -= 1
                self.OverlayData[self.z] = np.delete(self.OverlayData[self.z],-1,axis=0)
            self.ax.set_title(self.markpointsmodetitle+"\nMarked points in this plane: "+str(len(self.OverlayData[self.z])-1)+"\nMarked points in the stack: "+str(self.markedPointsN))
            self.update()

    def update(self):
        if len(self.dimensions) == 4: 
            self.im.set_data(self.X[self.ch ,self.z, ...])#/self.scale[self.ch])
        else:
            self.im.set_data(self.X[self.z, ...])#/self.scale[self.ch])
        if self.maxX*self.scale[self.ch] > self.im.norm.vmin:
            norm = colors.Normalize(vmin=self.im.norm.vmin, vmax=self.maxX*self.scale[self.ch])
            self.im.set_norm(norm)
        self.ax.set_xlabel('slice ' + str(self.z) + "   channel "+str(self.ch))
        self.im.set_cmap(self.Cmap[self.ch])
        
        try:
            if self.colors[self.ch] != 'red' and self.colors[self.ch] != 'magenta': 
                goodcolor='r'
            else:
                goodcolor='g'
            if(isinstance(self.OverlayData[0],list)):
                punti = self.OverlayData[self.ch][self.z].T
            else:
                punti = self.OverlayData[self.z].T
                labs = self.OverlayLabels[self.z] #TODO implement for different channels
            self.overlay.set_xdata(punti[0])
            self.overlay.set_ydata(punti[1])
            self.overlay.set_color(goodcolor)
            for t in self.OverlayTxtObj:
                t.remove()
            self.OverlayTxtObj = []
            for i in np.arange(len(labs)):
                ann = self.ax.annotate(labs[i], xy=(punti[0][i],punti[1][i]), xytext=(punti[0][i]+self.OverlayLabelDx,punti[1][i]+self.OverlayLabelDy), color=goodcolor)
                self.OverlayTxtObj.append(ann)
        except:
            pass        
        
        self.im.axes.figure.canvas.draw()
        
    def get_markedPoints(self):
        markedPoints = []
        for i in np.arange(self.slices):
            markedPoints.append(self.OverlayData[i][1:])
        return markedPoints


def hyperstack(A,order="cz",cmap='',colors=['blue','cyan','green','orange','red','magenta'],Overlay=[],OverlayLabels=[],plotNow=True):
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
    
    if order=="zc": A = np.swapaxes(A,0,1)
    # Check if current figure is empty, otherwise go to the next one
    cfn = plt.gcf().number
    if len(plt.gcf().axes) != 0: cfn += 1
    
    fig = plt.figure(cfn)
    ax = fig.add_subplot(111)
    tracker = IndexTracker(ax, A, cmap=cmap, colors=colors, Overlay=Overlay, OverlayLabels=OverlayLabels)
    fig.canvas.mpl_connect('key_press_event', tracker.onkeypress)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker.onbuttonpress)
    if plotNow==True:
        plt.show()
        
    return tracker   
