import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from PyQt5.QtCore import pyqtRemoveInputHook

def hyperstack(data, color = None, cmap = None,
               overlay = None, overlay_labels = None, hyperparam = None,
               refresh_time = 0.1, plot_now = True, live = False):
    '''
    '''
    
    # Get number of next available figure           
    cfn = plt.gcf().number
    if len(plt.gcf().axes) != 0: cfn += 1
    
    # Initialize plot
    fig = plt.figure(cfn)
    ax = fig.add_subplot(111)
    
    # Instantiate Hyperstack class
    iperpila = Hyperstack(ax, data, color = color, cmap = cmap, 
                 overlay = overlay, overlay_labels = overlay_labels, 
                 hyperparam = hyperparam, refresh_time = 0.1)
    
    # Bind the figure to the interactions events
    fig.canvas.mpl_connect('key_press_event', iperpila.onkeypress)
    fig.canvas.mpl_connect('scroll_event', iperpila.onscroll)
    fig.canvas.mpl_connect('button_press_event', iperpila.onbuttonpress)
    fig.canvas.mpl_connect('close_event', iperpila.onclose)
    
    # If no live mode has been requested and no plot_now, plot and block.
    if live == False and plot_now == True:
        plt.show()
    else:
        fig.show()
        
    return iperpila
    
def hyperstack_live_min_interface(data, stop, out, refresh_time=0.1): 
    #overlay, overlay_irrstrides
    # You'll have to add the overlay
    #overlay = mf.struct.irrarray(overlay_, irrStrides = [[3,17]], strideNames=["ch"])

    iperpila = hyperstack(data,live=True,refresh_time=refresh_time)

    while True:
        if iperpila.has_been_closed: break
        if stop[0]>0: 
            plt.close(iperpila.im.axes.figure)
            break
        iperpila.update(live=True)
        out[0:3] = iperpila.current_point
    
    return iperpila

def make_colormap(colors):
    '''Makes list of matplotlib color maps from black to colors.
    
    Parameters
    ----------
    colors: list or numpy array
        List of matplotlib-recognizable colors
        
    Returns
    -------
    cmaps: list
        List of matplotlib colormaps
    '''
    cmaps = []
    if not isinstance(colors, (list, tuple, np.ndarray)):
        colors = [colors]
    for q in np.arange(len(colors)):
        cmap = colors.LinearSegmentedColormap.from_list(
                                        'name', ['black',colors[q]])
        cmaps.append(cmap)
    
    return cmaps
    
def good_overlay_colors(colors):
    '''Makes list of good colors compatible with input colors.
    
    Parameters
    ----------
    colors: list or numpy array
        List of matplotlib-recognizable colors
    
    Returns
    -------
    good_colors: list
        List of good colors for overlays over input colors.
    
    '''
    good_colors = []
    if not isinstance(colors, (list, tuple, np.ndarray)):
        colors = [colors]
    for q in np.arange(len(colors)):
        if colors[q] not in ["red","r"]: 
            gc = "r"
        else:
            gc = "g"
        good_colors.append(gc)
    
def override_default_keys(new_keys_set):
    '''Disables default functions for given keys.
    
    Parameters
    ----------
    new_keys_set: list of strings
        List of keys for which default behavior has to be disabled.
        
    Returns
    -------
    None
    '''
    
    for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
                    
    return None

class Hyperstack():
    def_title = "Press h for instructions."
    instructions = \
            '\n**INSTRUCTIONS:\n'+\
            'Scroll/up-down to navigate in z\n'+\
            'A-D/left-right to change channel, W-S to change color scaling \n'+\
            'Ctrl+p for point labeling \n'+\
            'Ctrl+o for point marking (without labels)\n'+\
            'Ctrl+0 minimum of colorscale to 0 (to clip negative values)\n'+\
            'Ctrl+9 minimum of colorscale to original \n'+\
            'Ctrl+1 minimum of colorscale to 100 (for raw images with no '+\
            'binning) \n'+\
            'Ctrl+4 minimum of colorscale to 400 (for raw images with 2x2 '+\
            'binning: whole brain imager, MCW)'
    
    # Keys for which the default behavior has to be overwritten        
    new_keys_set = {'a', 'd', 'w', 's', 'up', 'down','left','right'}
    
    # Plot properties
    overlay_markersize = 1
    overlay_label_dx = 0
    overlay_label_dy = 0
    
    # Operating modes
    mode_select = False
    mode_label = False
    
    # Data
    current_point = np.zeros(3)
    
    # Utilites
    has_been_closed = False
    
    def __init__(self, ax, data, color = None, cmap = None, 
                 overlay = None, overlay_labels = None, hyperparam = None,
                 refresh_time = 0.1):
        '''Assumes data is [z,ch,y,x]
        '''
        
        # Don't really remember what this was for.
        pyqtRemoveInputHook()
        
        self.ax = ax
        self.ax.set_title(self.def_title)
        
        # Live update
        self.refresh_time = refresh_time
        
        # Prepare matplotlib to accept the new commands you set
        override_default_keys(self.new_keys_set)
        
        # Build colormaps
        if color is not None:
            self.cmaps = make_colormaps(color)
            self.good_overlay_colors = good_overlay_colors(colors)
        elif cmap is not None:
            if isinstance(cmap, (list,tuple,np.ndarray)):
                self.cmaps = cmap
            else:
                self.cmaps = [cmap]
            self.good_overlay_colors = ["k" for i in range(len(self.cmap))]
        else:
            self.cmaps = ["viridis"]
            self.good_overlay_colors = ["r"]
        self.n_colors = len(self.cmaps)     
        self.n_overlay_colors = len(self.good_overlay_colors)
        
        # The default data dimensionality is 4, i.e. [z, ch, y, x].
        # If the data is 2 or 3-dimensional, create a view of the array with
        # additional first axes, so that the syntax from now on can be for
        # the general 4-dimensional hyperstack case.
        # If the data is 3-dimensional, the default assumption is that the 0-th
        # axis is z, unless hyperparam == "c".
        # If the data is more than 4-dimensional, keep the input data as 
        # self.hyperdata, and make a 4-dimensional view of it. With a given 
        # keys, the user will be able to jump along those strides.
        n_dims = len(data.shape)
        print("n_dims",n_dims)
        if n_dims < 2: print("Throw exception, not an image")
        elif n_dims == 2: self.data = data[None,None,:,:]
        elif n_dims == 3: 
            if hyperparam is not "c":
                self.data = data[:,None,:,:]
            else:
                self.data = data[None,:,:,:]
                print(self.data.shape)
        elif n_dims == 4: self.data = data
        elif n_dims > 4:
            self.hyperdata = data
            zero = tuple([0 for i in range(n_dims-4)])
            self.data = data[zero]
        
        # Extract information about the data to be plotted
        self.dim = self.data.shape
        
        # Initialize current positions.
        self.z = self.dim[0]//2
        self.ch = 0
        
        self.scale = np.ones(self.dim[1])
        self.data_max = np.max(self.data)
        self.data_min = np.min(self.data)
        self.vmax = self.data_max
        self.vmin = self.data_min
        
        # overlay must be an irrarray with structure (ch)[i, coord]
        # This ensures that it's still contiguous in memory.
        # Postpone the hyperoverlay implementation.
        if overlay is not None:
            overlay_n_dims = len(overlay.first_index.keys())
            if overlay_n_dims == 1:
                self.overlay = overlay
                
            self.overlay_n_ch = len(overlay.first_index['ch'])-1
            '''elif n_dims > 2:
                self.hyperoverlay = overlay
                zero = tuple([0 for i in range(overlay_n_dims-2)])
                self.overlay = overlay[zero]'''
        else:
            self.overlay = None
            
        # Overlay labels
        # overlay_labels must be an irrarray with structure (ch)[i]
        if overlay_labels is not None:
            # Temporary
            self.overlay_labels = None
            
            # Make them the same shape as self.overlay, as above
            
            self.overlay_labels_dx = int(self.data.shape[-2]/100)
            self.overlay_labels_dy = int(self.data.shape[-1]/100)
        else:
            self.overlay_labels = None
        
        # Initialize plot
        self.im = self.ax.imshow(self.data[self.z,self.ch],
                                 interpolation="none",
                                 vmin=self.vmin, vmax=self.vmax)
        if self.overlay is not None:
            self.overlay_plot, = self.ax.plot(0,0,'o',
                                             markersize=self.overlay_markersize,
                                             c='k')
        if self.overlay is not None and self.overlay_labels is not None:
            ann = self.ax.annotate("",xy=(0,0),xytext=(0,0),color="k")
            self.overlay_labels_plot = [ann]
        
        
        self.update()
        
    def update(self, live = False):
    
        #######
        # Image
        #######
        self.im.set_data(self.data[self.z,self.ch])
        self.im.set_cmap(self.cmaps[self.ch % self.n_colors])
        self.ax.set_xlabel('slice '+str(self.z)+"   channel "+str(self.ch))
        # Adjust scale
        if self.data_max*self.scale[self.ch] > self.im.norm.vmin:
            norm = colors.Normalize(vmin=self.im.norm.vmin, 
                                    vmax=self.data_max*self.scale[self.ch])
            self.im.set_norm(norm)
                                 
        #########
        # Overlay
        #########
        if self.overlay is not None:
            # Extract the overlay points to be plotted in this channel
            ovrl = self.overlay(ch=self.ch%self.overlay_n_ch)
            
            # Extract the points to be plotted in this slice.
            if ovrl.shape[1] == 3:
                ovrl = ovrl[np.where(ovrl[:,0]==self.z%self.dim[0])]
            else:
                pass
                
            self.overlay_plot.set_xdata(ovrl[:,-1])
            self.overlay_plot.set_ydata(ovrl[:,-2])
            self.overlay_plot.set_color(
                    self.good_overlay_colors[self.ch%self.n_overlay_colors])
            self.overlay_plot.set_markersize(self.overlay_markersize)

        ################
        # Overlay labels
        ################
        if self.overlay is not None and self.overlay_labels is not None:
            # Remove previous labels
            for t in self.overlay_labels_plot:
                t.remove()
            
            # Extract the labels to be plotted in this channel    
            ovrl_labs = self.overlay_labels[self.ch%self.overlay.shape[-2]]
            
            # Extract the points to be plotted in this slice.
            if self.overlay.shape[-1] == 3:
                ovrl_labs = ovrl_labs[np.where(ovrl[0]==self.z%self.dim[-3])]
            else:
                pass
                
            # Add to the plot
            for i in np.arange(len(ovrl_labs)):
                ann = self.ax.annotate(ovrl_labs[i],
                                       xy=(ovrl[i,-1],ovrl[i,-2]),
                                       xytext=(ovrl[i,-1]+self.overlay_label_dx,
                                               ovrl[i,-2]+self.overlay_label_dy),
                                       color=self.good_overlay_colors[self.ch])
                self.overlay_labels_plot.append(ann)
           
        self.im.axes.figure.canvas.draw()
        
        if live:
            self.im.axes.figure.canvas.start_event_loop(self.refresh_time)
        
        
    def onscroll(self, event):
        if event.button == 'up':
            self.z = (self.z + 1) % self.dim[0]
        else:
            self.z = (self.z - 1) % self.dim[0]
        self.update()
    
    def onkeypress(self, event):
        if not self.mode_label:
            if event.key == 'h':
                print(self.instructions)
            elif event.key == 'd' or event.key == 'right':
                self.ch = (self.ch + 1) % self.dim[1]
            elif event.key == 'a' or event.key == 'left':
                self.ch = (self.ch - 1) % self.dim[1]
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
            elif event.key == 'ctrl+o': # Rename this to select
                self.mode_select = not self.mode_select
                if self.mode_select == True:
                    pass
                    #Do stuff
                else:
                    self.ax.set_title(self.defaultTitle)
            if event.key == 'ctrl+p': # Rename this to label
                self.mode_label = not self.mode_label
                if self.mode_label == True:
                    self.ax.set_title(
                                "Select points mode. "+\
                                "Press ctrl+p to switch back to normal mode.")
                else:
                    self.ax.set_title(self.defaultTitle)
                    
        self.update()
        
    def onbuttonpress(self, event):
        ix, iy = event.xdata, event.ydata
        if ix != None and iy != None:
            self.current_point[0] = self.z
            self.current_point[1] = iy
            self.current_point[2] = ix
            
    def onclose(self,event):
        self.has_been_closed = True
    
    def set_overlay_markersize(self,markersize):
        self.overlay_markersize = markersize
    
    def set_overlay_label_displacement(self,dx,dy):
        self.overlay_label_dx = dx
        self.overlay_label_dy = dy
        
