import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backend_tools import ToolToggleBase
from PyQt5.QtCore import pyqtRemoveInputHook
from PyQt5.QtWidgets import QAction
import string
import sys, termios

class LabelTool(ToolToggleBase):
    default_keymap = 'ctrl+p'
    description = 'Label'
    default_toggled = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def enable(self, *args):
        self.labeling = True

    def disable(self, *args):
        self.labeling = False

def hyperstack2(data, color = None, cmap = None,
               overlay = None, overlay_labels = None, manual_labels=None,
               hyperparam = None,
               side_views = True,
               plot_now = True, live = False,
               rgb = False,
               ext_event_callback = []):
    '''Function to use the Hyperstack class. 
    Parameters
    ----------
    data: numpy array
        The default interpretation of the dimensions is [z, c, y, x]. (c is
        channel). However, the initialization can accept also the following:
        - [z,y,x]
        - [c,y,x] with hyperparam set to "c"
        - [y,x]
    color: string or list of strings
        List of matplotlib-recognizable colors to create the black-to-color
        colormaps for the channels. The class is able to handle a number
        of colors different from the number of channels. Default: None.
    cmap: string list of strings
        List of matplotlib colormaps for the channels. The number of cmap
        can be different from the number of channels. Default: None. If 
        both cmap and color are None, the class defaults to viridis.
    overlay: irrarray
        Irregular array with irregular stride "ch", with overlay(ch=c)[i]
        giving z,y,x of the i-th overlay point in channel c. Default: None.
    overlay_label: irrarray
        Irregular array with the same structure as overlay, with
        overlay(ch=c)[i] giving the label of the i-th overlay point in c.
        Default: None
    hyperparam: string
        Specifier of the hyperparameters. Situation-dependent. 
        Default: None.
    plot_now: boolean
        If True (and live is False), the plot is shown in blocking mode.
        To use multiple hyperstacks, set plot_now to False (with live either
        not set or set to False) and store the value returned by this function
        to a variable, so that a reference to the Hyperstack instantiation is
        retained. After creating the other Hyperstacks, call plt.show() in the
        main scritp. Default: True.
    live: boolean
        If True, the plot is shown in non-blocking mode, so that the main script
        will keep running and the plot can be used to visualize data that is
        being updated live. To update the data, store the instance of the 
        Hyperstack class returned by this function and call its method
        update(live=True, refresh_time), where refresh_time is the time
        matplotlib will pause while updating the plot.
        
    Returns
    -------
    Instance of the Hyperstack class.
    '''
    
    # Get number of next available figure           
    cfn = plt.gcf().number
    if len(plt.gcf().axes) != 0: cfn += 1
    
    # Initialize plot
    fig = plt.figure(cfn)
    if side_views:
        fig.set_size_inches(10., 10., forward=True)
    
    # Instantiate Hyperstack class
    iperpila = Hyperstack(fig, data, color = color, cmap = cmap, 
                 overlay = overlay, overlay_labels = overlay_labels,
                 manual_labels=manual_labels, 
                 hyperparam = hyperparam, side_views = side_views,
                 rgb=rgb, ext_event_callback=ext_event_callback)
    
    # Bind the figure to the interactions events
    ev_conn_kp = fig.canvas.mpl_connect('key_press_event', iperpila.onkeypress)
    ev_conn_sc = fig.canvas.mpl_connect('scroll_event', iperpila.onscroll)
    ev_conn_bp = fig.canvas.mpl_connect('button_press_event', iperpila.onbuttonpress)
    fig.canvas.mpl_connect('axes_enter_event', iperpila.onaxisenter)
    fig.canvas.mpl_connect('figure_enter_event', iperpila.onfigureenter)
    fig.canvas.mpl_connect('figure_leave_event', iperpila.onfigureleave)
    fig.canvas.mpl_connect('close_event', iperpila.onclose)
    
    iperpila.event_connections["key_press_event"] = ev_conn_kp
    iperpila.event_connections["scroll_event"] = ev_conn_sc
    iperpila.event_connections["button_press_event"] = ev_conn_bp
    
    # Find the zoom and pan buttons and connect respective events
    actions = fig.canvas.manager.toolbar.actions()
    n_a = len(actions)
    for i_a in np.arange(n_a):
        if actions[i_a].text() == "Pan": 
            pan_action = fig.canvas.manager.toolbar.actions()[i_a]
            pan_action.toggled.connect(iperpila.onpantoggle)   
        if actions[i_a].text() == "Zoom": 
            zoom_action = fig.canvas.manager.toolbar.actions()[i_a]
            zoom_action.toggled.connect(iperpila.onzoomtoggle)   
            
    # Add buttons    
    label_action = QAction("Label", fig.canvas)
    label_action.setStatusTip("Labeling mode")
    label_action.triggered.connect(iperpila.onlabeltooltoggle)
    label_action.setCheckable(True)
    fig.canvas.manager.toolbar.addAction(label_action)
    
    select_action = QAction("Select", fig.canvas)
    select_action.setStatusTip("Selection mode")
    select_action.triggered.connect(iperpila.onselecttooltoggle)
    select_action.setCheckable(True)
    fig.canvas.manager.toolbar.addAction(select_action)
    
    zproj_action = QAction("z proj", fig.canvas)
    zproj_action.setStatusTip("z projection")
    zproj_action.triggered.connect(iperpila.onzprojtooltoggle)
    zproj_action.setCheckable(True)
    fig.canvas.manager.toolbar.addAction(zproj_action)
    
    overlay_action = QAction("overlay", fig.canvas)
    overlay_action.setStatusTip("Hide/show overlay")
    overlay_action.triggered.connect(iperpila.onoverlaytooltoggle)
    overlay_action.setCheckable(True)
    overlay_action.setChecked(True)
    fig.canvas.manager.toolbar.addAction(overlay_action)
    
    iperpila.extra_actions = {"label":label_action, "select": select_action,
                              "zproj": zproj_action, "overlay": overlay_action}
     
    # If no live mode has been requested and no plot_now, plot and block.
    if live == False and plot_now == True:
        plt.show()
    elif live == True:
        fig.show()
        
    return iperpila
    
def hyperstack_live_min_interface(data, stop, out, refresh_time=0.1): 
    '''Function to use the Hyperstack class in live mode with a minimal data
    interface, e.g. when embedded in C\C++.
    
    Parameters
    ----------
    data: numpy array
        The default interpretation of the dimensions is [z, c, y, x]. (c is
        channel). However, the initialization can accept also the following:
        - [z,y,x]
        - [c,y,x] with hyperparam set to "c"
        - [y,x]
    stop: numpy array of floats
        If stop[0]>0, the function stops and the hyperstack plot is closed.
        It uses floats instead of booleans to be more easily compatible, e.g.
        with DLL calls from LabView.
    out: numpy array
        Memory buffer to bring out results (e.g. the last selected point) to 
        the calling process. See implementation for what data is brought out.
    refresh_time: float
        Time matplotlib will pause while updating the plot.
    
    Returns
    -------
    1.0
    
    '''
    #overlay, overlay_irrstrides
    # You'll have to add the overlay
    #overlay = mf.struct.irrarray(overlay_, irrStrides = [[3,17]], strideNames=["ch"])

    iperpila = hyperstack2(data,live=True)

    while True:
        # When to stop 
        if iperpila.has_been_closed: break
        if stop[0]>0: 
            plt.close(iperpila.im.axes.figure)
            break
        
        # Update the plot
        try:
            iperpila.update(live=True,refresh_time=refresh_time)
        except:
            pass
        
        ##################################################################
        ## Extract information and pass it to the caller via the out array
        ##################################################################
        
        # Last clicked position
        out[0:4] = iperpila.current_point
        
        # Last pressed key for commands. Pass the letter index in the alphabet
        # in out[4] and whether ctrl was pressed in out[5]
        if iperpila.last_pressed_key.startswith("ctrl+"):
            last_key = iperpila.last_pressed_key[5:6]
            out[5] = 1.0
        elif iperpila.last_pressed_key.startswith("alt+"):
            last_key = iperpila.last_pressed_key[4:5]
            out[5] = 2.0
        elif iperpila.last_pressed_key == "control":
            pass
        elif iperpila.last_pressed_key == "alt":
            pass
        else:
            last_key = iperpila.last_pressed_key[0:1]
            out[5] = 0.0
        
        try:
            out[4] = string.ascii_lowercase.index(last_key)
        except:
            pass
    
    del iperpila
    return 1.0

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
            'Basic controls\n'+\
            '\tScroll mouse wheel or up-down arrows to navigate in z\n'+\
            '\ta-d or left-right arrows to change channel\n'+\
            '\tw-s to change color scaling \n'+\
            'Ctrl+ combinations control operating modes\n'+\
            '\tCtrl+p to label points (type label in terminal)\n'+\
            '\tCtrl+o to add points (left click adds, right click removes last)\n'+\
            'Alt+ combinations control visualization modes\n'+\
            '\tAlt+z to toggle between single plane and z-projection \n'+\
            '\tAlt+o to hide/display overlay \n'+\
            '\tAlt+a to toggle between \'equal\' and \'auto\' aspect ratio\n'+\
            '\tAlt+l to switch between live and static mode \n'+\
            '\tAlt+0 minimum of colorscale to 0 (to clip negative values)\n'+\
            '\tAlt+9 minimum of colorscale to original \n'+\
            '\tAlt+1 minimum of colorscale to 100 (for raw images with no \n'+\
            '\t    binning) \n'+\
            '\tAlt+4 minimum of colorscale to 400 (for raw images with 2x2 \n'+\
            '\t    binning: whole brain imager, MCW/neuropal in WBI room)\n'+\
            'plus any other command implemented in code that interfaces\n'+\
            '    with the hyperstack'
    
    # Keys for which the default behavior has to be overwritten        
    new_keys_set = {'a', 'd', 'w', 's', 'k', 'up', 'down','left','right'}
    
    # Plot properties
    overlay_markersize = 1
    overlay_label_dx = 0
    overlay_label_dy = 0
    rgb = False
    
    # Operating modes
    mode_select = False
    mode_label = False
    mode_typing = False
    active_axis = None
    figure_is_active = True
    
    # Data
    current_point = np.zeros(4)
    selected_points = np.zeros((0,4))
    last_clicked_point = np.zeros(2)
    
    # Utilites
    has_been_closed = False
    z_projection = False
    display_overlay = True
    side_views_slice = False
    last_pressed_key = 'a'
    instructions_shown = False
    message_shown = False
    pause_live_update = False
    new_action = False
    new_none_action = False
    last_action = ""
    
    def __init__(self, fig, data, color = None, cmap = None, 
                 overlay = None, overlay_labels = None, manual_labels=None,
                 hyperparam = None, side_views = True, rgb = False,
                 ext_event_callback = []):
        '''Class for plotting the hyperstack-visualization of data.
        Parameters
        ----------
        ax: matplotlib figure
            Initialized matplotlib figure in which to imshow the data.
        data: numpy array
            The default interpretation of the dimensions is [z, c, y, x]. (c is
            channel). However, the initialization can accept also the following:
            - [z,y,x]
            - [c,y,x] with hyperparam set to "c"
            - [y,x]
        color: string or list of strings
            List of matplotlib-recognizable colors to create the black-to-color
            colormaps for the channels. The class is able to handle a number
            of colors different from the number of channels. Default: None.
        cmap: string list of strings
            List of matplotlib colormaps for the channels. The number of cmap
            can be different from the number of channels. Default: None. If 
            both cmap and color are None, the class defaults to viridis.
        overlay: irrarray
            Irregular array with irregular stride "ch", with overlay(ch=c)[i]
            giving z,y,x of the i-th overlay point in channel c. Default: None.
        overlay_label: irrarray
            Irregular array with the same structure as overlay, with
            overlay(ch=c)[i] giving the label of the i-th overlay point in c.
            Default: None
        hyperparam: string
            Specifier of the hyperparameters. Situation-dependent. 
            Default: None.
        
        Returns
        -------
        None.
        '''
        self.event_connections = {}
        
        self.ext_event_callback = []
        if type(ext_event_callback) is not list: 
            ext_event_callback = [ext_event_callback]
        self.ext_event_callback = ext_event_callback
        
        # Don't really remember what this was for.
        pyqtRemoveInputHook()
        
        self.side_views = side_views
        if not self.side_views: 
            self.def_title = self.def_title[:-1] + " in terminal."
        self.rgb = rgb
        self.fig = fig
        if not side_views:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = self.fig.add_subplot(221)
            self.ax1 = self.fig.add_subplot(222, sharey=self.ax)
            self.ax2 = self.fig.add_subplot(223, sharex=self.ax)
        
        self.ax_title_locked = False    
        self.set_ax_title(self.def_title)
        
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
        if n_dims < 2: print("Throw exception, not an image")
        elif n_dims == 2: self.data = data[None,None,:,:]
        elif n_dims == 3: 
            if hyperparam != "c":
                self.data = data[:,None,:,:]
            else:
                self.data = data[None,:,:,:]
                print(self.data.shape)
        elif n_dims == 4: self.data = data
        elif n_dims == 5 and self.rgb == True: self.data = data
        elif n_dims > 4:
            self.hyperdata = data
            zero = tuple([0 for i in range(n_dims-4)])
            self.data = data[zero]
        
        self.data_live = self.data
        
        # Extract information about the data to be plotted
        self.dim = self.data.shape
        
        # Initialize current positions.
        self.z = self.dim[0]//2
        self.ch = 0
        
        self.z_descr = ""
        
        self.channel_titles = ["" for i in np.arange(self.dim[1])]
        if self.channel_titles[self.ch]!="":
            self.def_title_with_descr += "| "+self.channel_titles[self.ch]
            
        self.scale = np.zeros(self.dim[1])
        self.scale_min = np.zeros(self.dim[1])
        self.data_max = np.nanmax(self.data,axis=(0,2,3))
        self.data_min = np.nanmin(self.data,axis=(0,2,3))
        self.vmax = self.data_max[self.ch] if not self.rgb else 1
        self.vmin = self.data_min[self.ch] if not self.rgb else 1
        self.im_aspect = "auto"
        
        if self.side_views:
            self.data_max_side1 = np.nanmax(np.sum(self.data,axis=2),axis=(0,2))
            self.data_min_side1 = np.nanmin(np.sum(self.data,axis=2),axis=(0,2))
            self.data_max_side2 = np.nanmax(np.sum(self.data,axis=3),axis=(0,2))
            self.data_min_side2 = np.nanmin(np.sum(self.data,axis=3),axis=(0,2))
        
        # overlay must be an irrarray with structure (ch)[i, coord]
        # This ensures that it's still contiguous in memory.
        # Postpone the hyperoverlay implementation.
        if overlay is not None:
            overlay_n_dims = len(overlay.first_index.keys())
            if overlay_n_dims == 1:
                self.overlay = overlay
            self.overlay_n_ch = len(overlay.first_index['ch'])-1
            
            if manual_labels is None:
                self.manual_labels = [["" for i in np.arange(len(overlay(ch=ch_i)))] for ch_i in np.arange(self.overlay_n_ch)]
            else:
                self.manual_labels = manual_labels
            
            '''elif n_dims > 2:
                self.hyperoverlay = overlay
                zero = tuple([0 for i in range(overlay_n_dims-2)])
                self.overlay = overlay[zero]'''
        else:
            self.overlay = None
            
        # Overlay labels
        # overlay_labels must be an irrarray with structure (ch)[i]
        if overlay_labels is not None:
            if overlay_n_dims == 1:
                self.overlay_labels = overlay_labels
            
            if manual_labels is not None:
                labchn = len(manual_labels)
                for lablinei in np.arange(labchn):
                    lab = manual_labels[lablinei]
                    lablinen = len(lab)
                    for labi in np.arange(lablinen):
                        if lab[labi] != "":
                            self.overlay_labels(ch=lablinei)[labi] = lab[labi]     
            
            # Make them the same shape as self.overlay, as above
            
            self.overlay_labels_dx = int(self.data.shape[-2]/100)
            self.overlay_labels_dy = int(self.data.shape[-1]/100)
        else:
            self.overlay_labels = None
            
            
        # Initialize selected points
        self.selected_points = np.zeros((0,4))
        self.selected_points_labels = []
        
        # Initialize image plot
        self.im = self.ax.imshow(self.data[self.z,self.ch],
                                 interpolation="none", aspect=self.im_aspect,
                                 vmin=self.vmin, vmax=self.vmax)
                                 
        if self.side_views:
            # Initialize image plot for side views
            if not rgb:
                self.im2 = self.ax2.imshow(
                                    np.sum(self.data[:,self.ch,...],axis=1),
                                    interpolation="none",
                                    #vmin=self.vmin, vmax=self.vmax,
                                    aspect='auto')
                self.im1 = self.ax1.imshow(
                                    np.sum(self.data[:,self.ch,...],axis=2).T,
                                    interpolation="none",
                                    #vmin=self.vmin, vmax=self.vmax,
                                    aspect='auto')
            else:
                self.im2 = self.ax2.imshow(
                                    np.max(self.data[:,self.ch,...],axis=1),
                                    interpolation="none",
                                    #vmin=self.vmin, vmax=self.vmax,
                                    aspect='auto')
                self.im1 = self.ax1.imshow(
                                    np.swapaxes(np.max(self.data[:,self.ch,...],axis=2),0,1),
                                    interpolation="none",
                                    #vmin=self.vmin, vmax=self.vmax,
                                    aspect='auto')
                              
            # Initialize line cursors on main image/view
            self.im_axhline = self.im.axes.axhline(
               self.current_point[2],
               c=self.good_overlay_colors[self.ch%self.n_overlay_colors],
               lw=0.5)
            self.im_axvline = self.im.axes.axvline(
               self.current_point[3],
               c=self.good_overlay_colors[self.ch%self.n_overlay_colors],
               lw=0.5)
                     
            # Initialize line cursors on side views
            self.im2_axhline = self.im1.axes.axhline(
               self.z,
               c=self.good_overlay_colors[self.ch%self.n_overlay_colors])
            self.im2_axvline = self.im1.axes.axvline(
               self.current_point[2],
               c=self.good_overlay_colors[self.ch%self.n_overlay_colors])
            self.im1_axhline = self.im2.axes.axhline(
               self.z,
               c=self.good_overlay_colors[self.ch%self.n_overlay_colors])
            self.im1_axvline = self.im2.axes.axvline(
               self.current_point[3],
               c=self.good_overlay_colors[self.ch%self.n_overlay_colors])
        
        # Initialize overlay plot
        if self.overlay is not None:
            self.overlay_plot, = self.ax.plot(0,0,'o',
                                             markersize=self.overlay_markersize,
                                             c='k')
        # Initialize overlay labels plot
        if self.overlay is not None and self.overlay_labels is not None:
            ann = self.ax.annotate(".",xy=(0,0),xytext=(0,0),color="k")
            self.overlay_labels_plot = [ann]
            
            
        # Initialize selected points plot
        self.mode_select_plot, = self.ax.plot(
                    [],'x',
                    c=self.good_overlay_colors[self.ch%self.n_overlay_colors])
        
        # Initialize selected point labels plot
        ann_sel = self.ax.annotate(".",xy=(0,0),xytext=(0,0),color="k")
        self.selected_points_labels_plot = [ann_sel]
        
        self.fig.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
        self.update()
        
    def update(self, live = False, refresh_time=0.1, *args, **kwargs):
        ############
        # Main image
        ############
        # Channel description
        if self.channel_titles[self.ch]!="":
            ch_descr = self.channel_titles[self.ch]
        else:
            ch_descr = str(self.ch)
        
        if not self.z_projection:
            self.im.set_data(self.data[self.z,self.ch])
            self.ax.set_xlabel('x (z = '+str(self.z)+")   channel "+ch_descr+\
                                " | "+self.z_descr)
        else:
            if not self.rgb:
                self.im.set_data(np.sum(self.data[:,self.ch],axis=0))
            else:
                self.im.set_data(np.max(self.data[:,self.ch],axis=0))
            self.ax.set_xlabel("x (z projection) of channel "+ch_descr)
        self.ax.set_ylabel("y")
        
        dx = abs(self.ax.get_xlim()[1]-self.ax.get_xlim()[0])
        dy = abs(self.ax.get_ylim()[1]-self.ax.get_ylim()[0])
        self.ax.set_aspect(self.im_aspect)#dx/dy)#auto
        
        if not self.rgb:    
            self.im.set_cmap(self.cmaps[self.ch % self.n_colors])
            # Adjust scale
            desired_max = self.data_max[self.ch]-(self.data_max[self.ch]-self.data_min[self.ch])*self.scale[self.ch]
            desired_min = self.data_min[self.ch]+(self.data_max[self.ch]-self.data_min[self.ch])*self.scale_min[self.ch]
            #if self.data_max*self.scale[self.ch] > self.data_min:
            if desired_max > desired_min:#self.im.norm.vmin:
                norm = colors.Normalize(vmin=desired_min, 
                                        vmax=desired_max)
                self.im.set_norm(norm)
        
        ########################
        # Side views, if present
        ########################
        if self.side_views:
            if not self.side_views_slice:
                if not self.rgb:
                    self.im2.set_data(np.sum(self.data[:,self.ch,...],axis=1))
                    self.im1.set_data(np.sum(self.data[:,self.ch,...],axis=2).T)
                else:
                    self.im2.set_data(np.max(self.data[:,self.ch,...],axis=1))
                    self.im1.set_data(np.swapaxes(np.max(self.data[:,self.ch,...],axis=2),0,1))
                self.im2.axes.set_xlabel("x (y projection)")
                self.im2.axes.set_ylabel("z")
                self.im1.axes.set_ylabel("y (x projection)")
                self.im1.axes.set_xlabel("z")
            else:
                y = int(self.current_point[2])
                x = int(self.current_point[3])
                self.im2.set_data(self.data[:,self.ch,y,:])
                self.im2.axes.set_xlabel("x (y = "+str(y)+")")
                self.im2.axes.set_ylabel("z")
                if not self.rgb:
                    self.im1.set_data(self.data[:,self.ch,:,x].T)
                else:
                    self.im1.set_data(np.swapaxes(self.data[:,self.ch,:,x],0,1))
                self.im1.axes.set_ylabel("y (x = "+str(x)+")")
                self.im1.axes.set_xlabel("z")
            
            ## Update line cursors
            y = self.current_point[2]
            x = self.current_point[3]
            z = self.z
            # Main image
            self.im_axhline.set_data([0,self.dim[2]],[y,y])
            self.im_axvline.set_data([x,x],[0,self.dim[3]])
            self.im2_axvline.set_data([z,z],[0,self.dim[3]])
            self.im2_axhline.set_data([0,self.dim[0]],[y,y])
            self.im1_axhline.set_data([0,self.dim[2]],[z,z])
            self.im1_axvline.set_data([x,x],[0,self.dim[0]])
                
            if not self.rgb:
                # Adjust scale
                norm1 = colors.Normalize(vmin=self.data_min_side1[self.ch],
                                         vmax=self.data_max_side1[self.ch])
                norm2 = colors.Normalize(vmin=self.data_min_side2[self.ch],
                                         vmax=self.data_max_side2[self.ch])
                self.im1.set_norm(norm1)
                self.im2.set_norm(norm2)
            
            self.im1.axes.set_xlim(0,max(self.dim[0]-1,0.5))
            self.im1.axes.set_ylim(self.im.axes.get_ylim())
            self.im2.axes.set_xlim(self.im.axes.get_xlim())
            self.im2.axes.set_ylim(max(self.dim[0]-1,0.5),0)
                                 
        #########
        # Overlay
        #########
        if self.overlay is not None and self.display_overlay:
            # Extract the overlay points to be plotted in this channel
            ovrl = self.overlay(ch=self.ch%self.overlay_n_ch)
            
            # Extract the points to be plotted in this slice.
            if ovrl.shape[1] == 3 and not self.z_projection:
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
        if self.overlay is not None and self.overlay_labels is not None and self.display_overlay:
            # Remove previous labels
            for t in self.overlay_labels_plot:
                try:
                    t.remove()
                except:
                    pass
            
            # Extract the labels to be plotted in this channel    
            ovrl_labs = self.overlay_labels(ch=self.ch%self.overlay_n_ch)
            
            # Extract the points to be plotted in this slice.
            if self.overlay.shape[-1] == 3 and not self.z_projection:
                ovrl = self.overlay(ch=self.ch%self.overlay_n_ch)
                ovrl_labs = ovrl_labs[np.where(ovrl[:,0]==self.z%self.dim[0])]
                ovrl = ovrl[np.where(ovrl[:,0]==self.z%self.dim[0])]
                
            else:
                pass
                
            # Add to the plot
            self.overlay_labels_plot = []
            for i in np.arange(len(ovrl_labs)):
                ann = self.ax.annotate(ovrl_labs[i].lstrip("0"),
                                       xy=(ovrl[i,-1],ovrl[i,-2]),
                                       xytext=(ovrl[i,-1]+self.overlay_label_dx,
                                               ovrl[i,-2]+self.overlay_label_dy),
                                       color=self.good_overlay_colors[
                                                self.ch%self.n_overlay_colors],
                                       fontsize=8)
                self.overlay_labels_plot.append(ann)
                
        #################
        # SELECTED POINTS
        #################
        if self.selected_points.shape[0] > 0:
            sel_pts = self.selected_points[np.where(
                                (self.selected_points[:,0] == self.z)*(self.selected_points[:,1] == self.ch))]
            
            self.mode_select_plot.set_xdata(sel_pts[:,-1])
            self.mode_select_plot.set_ydata(sel_pts[:,-2])
            
            self.mode_select_plot.set_color(
                    self.good_overlay_colors[self.ch%self.n_overlay_colors])
            self.mode_select_plot.set_markersize(self.overlay_markersize)
        elif self.mode_select and self.selected_points.shape[0] == 0:
            self.mode_select_plot.set_xdata(-1)
            self.mode_select_plot.set_ydata(-1)
            
        # Selected points labels
        if self.display_overlay:
            # Remove previous labels
            for t in self.selected_points_labels_plot:
                try:
                    t.remove()
                except:
                    pass
            
            indices = np.where((self.selected_points[:,0]==self.z)*(self.selected_points[:,1]==self.ch))
            slpts = self.selected_points[indices]
            slpts_labs = np.array(self.selected_points_labels)[indices]
            
            # Add to the plot
            self.selected_points_labels_plot = []
            for i in np.arange(len(slpts_labs)):
                ann = self.ax.annotate(slpts_labs[i].lstrip("0"),
                                       xy=(slpts[i,-1],ovrl[i,-2]),
                                       xytext=(slpts[i,-1]+self.overlay_label_dx,
                                               slpts[i,-2]+self.overlay_label_dy),
                                       color=self.good_overlay_colors[
                                                self.ch%self.n_overlay_colors],
                                       fontsize=8)
                self.selected_points_labels_plot.append(ann)
        
        ########
        # Redraw
        ########
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        self.set_last_action(None)
        
        if live:
            pass#self.fig.canvas.start_event_loop(refresh_time)
        
        
    def onscroll(self, event):
        self.fig.canvas.mpl_disconnect(self.event_connections["scroll_event"])
        if event.button == 'up':
            self.z = (self.z + 1) % self.dim[0]
        else:
            self.z = (self.z - 1) % self.dim[0]
        
        self.update()
        ev_conn_sc = self.fig.canvas.mpl_connect('scroll_event',self.onscroll)
        self.event_connections["scroll_event"] = ev_conn_sc
    
    def onkeypress(self, event):
        self.last_pressed_key = event.key
        
        if not self.mode_typing: # To prevent actions while typing labels
            if event.key == 'h':
                self.show_instructions()
            elif event.key == 'd' or event.key == 'right':
                self.ch = (self.ch + 1) % self.dim[1]
            elif event.key == 'a' or event.key == 'left':
                self.ch = (self.ch - 1) % self.dim[1]
            elif event.key == 's':
                self.scale[self.ch] = max(self.scale[self.ch]-0.05,0.)#min(self.scale[self.ch]*1.3, 1.0)
            elif event.key == 'w':
                self.scale[self.ch] = min(self.scale[self.ch]+0.05,1.)#max(self.scale[self.ch]*0.7, 0.0001)
            elif event.key == 'i':
                self.scale_min[self.ch] = min(self.scale_min[self.ch]+0.05,1.)
            elif event.key == 'k':
                self.scale_min[self.ch] = max(self.scale_min[self.ch]-0.05,0.)
            elif event.key == 'up':
                self.z = (self.z + 1) % self.dim[0]
            elif event.key == 'down':
                self.z = (self.z - 1) % self.dim[0]
            elif event.key == 'alt+0':
                self.im.norm.vmin = 0
            elif event.key == 'alt+9':
                self.im.norm.vmin = self.data_min
            elif event.key == 'alt+1':
                self.im.norm.vmin = 100
            elif event.key == 'alt+4':
                self.im.norm.vmin = 400
            elif event.key == 'alt+l':
                # Toggle live update
                self.pause_live_update = not self.pause_live_update
                if self.pause_live_update:
                    self.data_live = self.data
                    self.data = np.copy(self.data)
                else:
                    self.data = self.data_live
            elif event.key == 'alt+z':
                self.toggle_z_projection()
            elif event.key == 'alt+o':
                self.toggle_overlay()                
            elif event.key == 'alt+a':
                self.im_aspect = 'auto' if self.im_aspect == "equal" else "equal"
                
            elif event.key == 'ctrl+o': 
                self.toggle_mode_select()
                    
            elif event.key == 'ctrl+p' and self.overlay is not None:
                self.toggle_mode_label()
        
        #elif event.key == 'ctrl+p' and self.overlay is not None:
        #    self.mode_label = not self.mode_label
        #    if self.mode_label == False: self.ax.set_title(self.def_title)          
                    
        self.update()
        
    def onbuttonpress(self, event):
        if self.fig.canvas.toolbar.mode == '':
            ix, iy = event.xdata, event.ydata
            self.last_clicked_point[0] = iy
            self.last_clicked_point[1] = ix
            
            if self.active_axis is self.ax and self.figure_is_active:
                if ix != None and iy != None:
                    self.current_point[0] = self.z
                    self.current_point[1] = self.ch
                    self.current_point[2] = iy % self.dim[2]
                    self.current_point[3] = ix % self.dim[3]
                    
                if self.mode_select and event.button == 1 and not event.dblclick:
                    # Add point
                    self.selected_points = np.append(
                                            self.selected_points,
                                            [self.current_point], axis=0)
                    self.selected_points_labels.append("")                        
                    self.set_last_action("point_added")
                elif self.mode_select and event.button == 3 and not event.dblclick:
                    # Delete last point
                    if self.selected_points.shape[0] > 0:
                        self.selected_points = np.delete(
                                                self.selected_points,
                                                -1,axis=0)
                        self.selected_points_labels.pop(-1)
                        self.set_last_action("point_deleted")
                                                
                elif self.mode_label and event.button == 1 and not event.dblclick:
                    self.fig.canvas.mpl_disconnect(self.event_connections["button_press_event"])
                    self.mode_typing = True
                    cl, in_selected = self.get_closest_point(include_selected_points=True)
                    if not in_selected:
                        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                        lab = input("Label for pt "+str(cl)+" (blank to keep "+self.manual_labels[cl[0]][cl[1]]+"): ")
                        if lab!="": self.manual_labels[cl[0]][cl[1]] = lab
                        else: lab = self.manual_labels[cl[0]][cl[1]]
                        self.overlay_labels(ch=cl[0])[cl[1]] = lab
                        self.set_last_action("point_labeled:"+",".join([str(_cl) for _cl in cl])+"-"+lab)
                    else:
                        i_in_selected = cl[1]
                        cl[1] += len(self.overlay_labels(ch=cl[0]))
                        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                        lab = input("\rLabel for pt "+str(cl)+" (blank to keep "+self.selected_points_labels[i_in_selected]+"): ")
                        if lab!="": self.selected_points_labels[i_in_selected] = lab
                        self.set_last_action("point_labeled:"+",".join([str(_cl) for _cl in cl])+"-"+lab)
                    
                    self.mode_typing = False
                    ev_conn_kp = self.fig.canvas.mpl_connect('button_press_event',self.onbuttonpress)
                    self.event_connections["button_press_event"] = ev_conn_kp
                    
                self.update()
        #elif self.fig.canvas.toolbar.mode != '' and (self.mode_label or self.mode_select):
        #    print("Unselect toolbar button (zoom/pan) to label or select points.")
                
    def onaxisenter(self,event):
        if self.figure_is_active:
            self.active_axis=event.inaxes
    
    def onfigureenter(self,event):
        self.figure_is_active = (event.canvas.figure.number == self.fig.number)
        
    def onfigureleave(self,event):
        self.figure_is_active = not (event.canvas.figure.number == self.fig.number)
    
    def onclose(self,event):
        self.has_been_closed = True
        
    def onpantoggle(self,event):
        if event:
            self.ax.set_title("Image pan selected. Unselect it to label/select points.")
            self.ax_title_locked = True
        elif self.fig.canvas.toolbar.mode=='pan/zoom':
            # Only reset the actual title if no other tool is active.
            # Otherwise, when pan is unselected by clicking on zoom (or vice 
            # versa), this will undo the title setting of the other callback.
            self.ax_title_locked = False
            self.set_ax_title(self.current_title) 
        self.update()
        
    def onzoomtoggle(self,event):
        if event:
            self.ax.set_title("Image zoom selected. Unselect it to label/select points.")
            self.ax_title_locked = True
        elif self.fig.canvas.toolbar.mode=='zoom rect':
            # Only reset the actual title if no other tool is active.
            # Otherwise, when pan is unselected by clicking on zoom (or vice 
            # versa), this will undo the title setting of the other callback.
            self.ax_title_locked = False
            self.set_ax_title(self.current_title) 
        self.update()
    
    def onlabeltooltoggle(self,event):
        self.toggle_mode_label(event)
        
    def onselecttooltoggle(self,event):
        self.toggle_mode_select(event)
        
    def onzprojtooltoggle(self,event):
        self.toggle_z_projection(event)
        
    def onoverlaytooltoggle(self,event):
        self.toggle_overlay(event)

    def create_additional_plot(self):
        try:
            self.ax3
        except:
            self.ax3 = self.fig.add_subplot(224)
            self.additional_plot, = self.ax3.plot(0,0,'-')
            
    def update_additional_plot(self,A,B=None,label=None):
        if B is None: 
            Y = A
            X = np.arange(A.shape[0])
        else:
            X = A
            Y = B
            
        try:
            self.ax3
        except:
            self.create_additional_plot()
            
        self.additional_plot.set_xdata(X)
        self.additional_plot.set_ydata(Y)
        
        self.ax3.set_xlim(np.min(X),np.max(X))
        try:
            self.ax3.set_ylim(np.min(Y),np.max(Y))
        except:
            pass
        
        if label is not None:
           self.additional_plot.set_label(str(label))
           self.ax3.legend()
           
    def clear_additional_plot(self):
        self.additional_plot.set_xdata([])
        self.additional_plot.set_ydata([])
    
    def remove_additional_plot(self):
        try:
            self.fig.delaxes(self.ax3)
        except:
            pass
    
    def set_data(self,data):
        self.data = data
        n_dims = len(data.shape)
        if n_dims < 2: print("Throw exception, not an image")
        elif n_dims == 2: self.data = data[None,None,:,:]
        elif n_dims == 3: 
            if hyperparam != "c":
                self.data = data[:,None,:,:]
            else:
                self.data = data[None,:,:,:]
                print(self.data.shape)
        elif n_dims == 4: self.data = data
        elif n_dims > 4:
            self.hyperdata = data
            zero = tuple([0 for i in range(n_dims-4)])
            self.data = data[zero]
        
        self.data_live = self.data
        
        # Extract information about the data to be plotted
        self.dim = self.data.shape
        
        # Initialize current positions.
        self.z = self.dim[0]//2
        self.ch = 0
        
        self.scale = np.ones(self.dim[1])
        self.data_max = np.nanmax(self.data,axis=(0,2,3))
        self.data_min = np.nanmin(self.data,axis=(0,2,3))
        self.vmax = self.data_max[self.ch]
        self.vmin = self.data_min[self.ch]
        
        if self.side_views:
            self.data_max_side1 = np.max(np.sum(self.data,axis=2),axis=(0,2))
            self.data_min_side1 = np.min(np.sum(self.data,axis=2),axis=(0,2))
            self.data_max_side2 = np.max(np.sum(self.data,axis=3),axis=(0,2))
            self.data_min_side2 = np.min(np.sum(self.data,axis=3),axis=(0,2))
    
    def set_overlay_markersize(self,markersize):
        self.overlay_markersize = markersize
    
    def set_overlay_label_displacement(self,dx,dy):
        self.overlay_label_dx = dx
        self.overlay_label_dy = dy
        
    def set_ax_title(self,title):
        if not self.ax_title_locked:
            self.ax.set_title(title)
        self.current_title = title
        
    def toggle_z_projection(self,active=None):
        if active is None:
            self.z_projection = not self.z_projection
        else:
            self.z_projection = active
        self.side_views_slice = self.z_projection
        
        # Recalculate data_max and data_min
        if not self.z_projection:
            self.data_max = np.nanmax(self.data,axis=(0,2,3))
            self.data_min = np.nanmin(self.data,axis=(0,2,3))
            self.data_max_side1 = np.nanmax(np.sum(self.data,axis=3),axis=(0,2))
            self.data_min_side1 = np.nanmin(np.sum(self.data,axis=3),axis=(0,2))
            self.data_max_side2 = np.nanmax(np.sum(self.data,axis=2),axis=(0,2))
            self.data_min_side2 = np.nanmin(np.sum(self.data,axis=2),axis=(0,2))
        else:
            self.data_max_side1 = self.data_max
            self.data_max_side2 = self.data_max
            self.data_min_side1 = self.data_min
            self.data_min_side2 = self.data_min
            self.data_max = np.nanmax(np.sum(self.data,axis=0),axis=(1,2))
            self.data_min = np.nanmin(np.sum(self.data,axis=0),axis=(1,2))
            
        if "zproj" in self.extra_actions.keys():
            self.extra_actions["zproj"].setChecked(self.z_projection)
            
    def toggle_overlay(self,active=None):
        if active is None:
            self.display_overlay = not self.display_overlay
        else:
            self.display_overlay = active
        if not self.display_overlay: self.hide_overlay()
        
        if "overlay" in self.extra_actions.keys():
            self.extra_actions["overlay"].setChecked(self.display_overlay)
        
    def toggle_mode_label(self,active=None):
        # Label overlay. Enable only if the overlay is not None 
        if active is None:
            self.mode_label = not self.mode_label
        else:
            self.mode_label = active
            
        if self.mode_label == True:
            self.toggle_mode_select(False)
            self.set_ax_title(
                        "Labeling overlay. "+\
                        "ctrl+p to switch to normal mode.\n"+\
                        "After clicking a point, insert input on terminal")
        else: self.set_ax_title(self.def_title)
        
        if "label" in self.extra_actions.keys():
            self.extra_actions["label"].setChecked(self.mode_label)
        
    def toggle_mode_select(self,active=None):
        if active is None:
            self.mode_select = not self.mode_select
        else:
            self.mode_select = active
            
        if self.mode_select == True: 
            self.toggle_mode_label(False)
            self.set_ax_title("Selecting points.")
        else:self.set_ax_title(self.def_title)
            
        if "select" in self.extra_actions.keys():
            self.extra_actions["select"].setChecked(self.mode_select)   
        
    def hide_overlay(self):
        if self.overlay is not None:
            self.overlay_plot.set_xdata([])
            self.overlay_plot.set_ydata([])
        if self.overlay_labels is not None:
            for t in self.overlay_labels_plot:
                try:
                    t.remove()
                except:
                    pass
        
    def show_instructions(self):
        self.instructions_shown = not self.instructions_shown
        if self.instructions_shown:
            if not self.side_views:
                print(self.instructions)
            else:
                self.im.axes.set_title("Press h to hide instructions")
                self.instructions_plot = self.fig.text(
                                        0.55,0.18,self.instructions,fontsize=10)
        else:
            self.im.axes.set_title(self.def_title_with_descr)
            self.instructions_plot.remove()
            
    def toggle_message(self, message=None):
        self.message_shown = not self.message_shown
        if self.message_shown: 
            if message is not None and self.side_views:
                self.message_plot = self.fig.text(0.5,0.1,message,fontsize=10)
        else:
            self.message_plot.remove()
            
    def get_current_point(self):
        return self.current_point
        
    def get_last_clicked_point(self):
        return self.last_clicked_point
        
    def get_active_axis(self):
        return self.active_axis
        
    def get_closest_point(self,include_selected_points=False):
        '''Get overlay point closest to last click.
        
        Returns
        -------
        closest_point: np.array
            [ch, i] overlay point i in channel ch.
        '''
        if self.overlay.shape[-1] == 3:
            pt = np.array([self.current_point[0],
                           self.current_point[2],
                           self.current_point[3]])
        else:
            pt = np.array([self.current_point[2],
                           self.current_point[3]])
        
        ovrl = self.overlay(ch=self.ch%self.overlay_n_ch)
        
        if len(ovrl)>0:
            D = np.sum(np.power(pt - ovrl,2),axis=-1)
            match = np.argmin(D)
            Dmin = D[match]
        else:
            match = 0
            
        in_selected = False
        if include_selected_points and self.selected_points.shape[0]>0:
            pt = np.array([self.current_point[2],
                            self.current_point[3]])
            slpts = self.selected_points[np.where(self.selected_points[:,1]==self.ch)]
            D2 = np.sum(np.power(pt-slpts[:,2:],2),axis=-1)
            match2 = np.argmin(D2)
            D2min = D2[match2]
            
            if D2min<Dmin:
                match = match2
                in_selected = True
        
        closest_point = np.array([self.ch%self.overlay_n_ch,match])
        
        if not include_selected_points: return closest_point
        else: return closest_point, in_selected
        
    def get_selected_points(self):
        return self.selected_points
    
    def get_manual_labels(self):
        return self.manual_labels
        
    def set_last_action(self,action):
        if action is not None:
            self.last_action = action
            self.new_action = True
        else:
            self.new_none_action = True
        for clbck in self.ext_event_callback:
            clbck(action)
        
    def get_last_action(self):
        self.new_action = False
        return self.last_action
