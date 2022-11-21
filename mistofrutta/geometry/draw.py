import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors
import shapely.geometry as geom
import scipy.ndimage
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import PolygonSelector
from matplotlib.widgets import Slider

class line:
    def __init__(self, image, verbose=True, custom_instructions="", yx=False,
                 plot_now = True):
    
        self.X = []
        self.Y = []
        self.yx = yx
    
        # Print instructions for the user
        if verbose:
            print("INSTRUCTIONS\n"+\
                ""+custom_instructions+"\n"+\
                "Click to add a point. Right-click to delete the last one. "+\
                "Ignore the first point in the corner of the image. " +\
                "Once you're done clicking, simply close the figure window.\n")    
        
        # Create the figure and plot the image
        cfn = plt.gcf().number
        if len(plt.gcf().axes) != 0: cfn += 1
        # Initialize plot
        fig = plt.figure(cfn)
        ax1 = fig.add_subplot(111)
        if yx:
            ax1.imshow(image)
        else:
            ax1.imshow(np.swapaxes(image,0,1))
        self.line, = ax1.plot(0,0,'*k')
        
        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)
        
        # Start displaying the plot
        if plot_now: plt.show()

    # Define callback function to be called when there is a mouse click
    def __call__(self, event):
        if plt.get_current_fig_manager().toolbar.mode == '':
            ix, iy = event.xdata, event.ydata
            # If left click, add point, otherwise delete the last one.
            if (event.button==1):
                self.X.append(ix)
                self.Y.append(iy)
            else:
                self.X.pop(-1)
                self.Y.pop(-1)
            
            self.line.set_data(self.X, self.Y)
            self.line.figure.canvas.draw()

    def getLine(self):
        # Convert to np.array and return    
        if self.yx:
            return np.array([self.Y,self.X]).T
        else:
            return np.array([self.X,self.Y]).T #[::-1]
        
    def get_line(self):
        return self.getLine()
        
        
class rectangle:
    def __init__(self, image, multiple=False, title="Select a rectangle"):
    
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        
        self.multiple = multiple
        self.rectangles = []
        
        self.Image = image
    
        # Print instructions for the user
        if self.multiple: 
            print("Multiple rectangles. ")
            title = "Select multiple rectangles"
        
        print("Press h for instructions\n")
        self.instructions = "\nINSTRUCTIONS TO DRAW THE RECTANGLE\n\n"+\
            "Click and drag the pointer.\n"+\
            "You can resize the rectangle dragging the markers at the "+\
            "corners, and move it dragging the marker at the center.\n"+\
            "You can delete the rectangle clicking anywhere on the image.\n"+\
            "Once you're done, simply close the figure window."  
            
        # Print additional instructions in case multiple is true
        if self.multiple:
            self.instructions += "\n\nMULTIPLE RECTANGLES\n\n"+\
                "You can save multiple rectangles: add the drawn rectangle "+\
                "pressing \"a\" and delete the last one pressing \"d\".\n" +\
                "After you add a rectangle, you can proceed directly to draw "+\
                "the next one.\n\n"
        
        # Create the figure
        self.fig = plt.figure(1)
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_title(title)
        
        # Set a custom viridis color map useful for thresholding
        self.cmap = plt.cm.viridis
        cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        cmaplist[0] = (1.,0.,0.,1.)
        cmaplist[-1] = (1.,0.5,0.,1.)
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, self.cmap.N)
        
        # Plot the image
        self.ax1.imshow(self.Image.T,cmap=self.cmap,aspect="auto")
        
        # Prepare the rectangle selector
        rectprops = dict(facecolor='red', edgecolor = 'white',
                 alpha=1.0, fill=False, lw=2)
        self.rs = RectangleSelector(self.ax1, self.line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True, rectprops=rectprops)
                       
        plt.connect('key_press_event', self.keyboardCallback)
        
        # Add slider to set threshold
        self.vmin = np.min(self.Image)
        self.vmax = np.max(self.Image)
        axThresholdSlider = plt.axes([0.25, 0.02, 0.65, 0.015], facecolor='lightgoldenrodyellow')
        self.thresholdSlider = Slider(axThresholdSlider, 'Threshold', self.vmin*0.95, self.vmax, valinit=self.vmin)
        self.thresholdSlider.on_changed(self.updateThreshold)
        
        plt.show()


    def line_select_callback(self,eclick,erelease):
        return 1
   
    def updateThreshold(self, val):
        if abs(1.0-val/self.vmin) > 0.05:
            self.vmin = val
            self.ax1.imshow(self.Image.T, vmin=val, cmap=self.cmap)
            self.fig.canvas.draw()
        
    
    def keyboardCallback(self,event):
        if event.key in ['A', 'a']:
            if self.multiple:
                self.x1 = int(np.min(self.rs.corners[0]))
                self.x2 = int(np.max(self.rs.corners[0]))
                self.y1 = int(np.min(self.rs.corners[1]))
                self.y2 = int(np.max(self.rs.corners[1]))
                self.rectangles.append(np.array([[self.x1,self.y1],
                                                [self.x2,self.y2]]))
                print("Adding rectangle\t"+str(int(self.x1))+" "+\
                                          str(int(self.x2))+" "+\
                                          str(int(self.y1))+" "+\
                                          str(int(self.y2))+" ")
            else:
                print("You need to initialize the rectangle selector with" +\
                      "multiple=True to save multiple rectangles.")
            
        elif event.key in ['D', 'd']:
            if self.multiple:
                if len(self.rectangles) > 0:
                    p = self.rectangles[-1]
                    print("Removing last rectangle\t"+str(int(p[0,0]))+" "+\
                                                      str(int(p[0,1]))+" "+\
                                                      str(int(p[1,0]))+" "+\
                                                      str(int(p[1,1]))+" ")
                    self.rectangles.pop()
                else:
                    print("There are no rectangles to delete.")
                
            else:
                print("You need to initialize the rectangle selector with" +\
                      "multiple=True to save multiple rectangles.")
        elif event.key in ['H', 'h']:
            print(self.instructions)
                

        
    def getRectangle(self,mode="coordinates"):
        # Returns a single rectangle if self.multiple is false.
        # Otherwise, it returns an array of rectangles.
        
        if not self.multiple:
            self.x1 = int(np.min(self.rs.corners[0]))
            self.x2 = int(np.max(self.rs.corners[0]))
            self.y1 = int(np.min(self.rs.corners[1]))
            self.y2 = int(np.max(self.rs.corners[1]))
            
            if mode=="coordinates":
                return np.array([[self.x1,self.y1],[self.x2,self.y2]])
            elif mode=="slice":
                return tuple([slice(*i) for i in zip([self.x1,self.y1],[self.x2,self.y2])])
            elif mode=="slice_full":
                return tuple([slice(*i) for i in zip([None,None,self.x1,self.y1],[None,None,self.x2,self.y2])])
        else:
            if mode=="coordinates":
                return np.array(self.rectangles)
            elif mode=="slice":
                slices = []
                for r in self.rectangles:
                    slices.append(tuple([slice(*i) for i in zip([r[0,0],r[0,1]], [r[1,0],r[1,1]])]))
                return slices
            elif mode=="slice_full":
                slices = []
                for r in self.rectangles:
                    slices.append(tuple([slice(*i) for i in zip([None, None, r[0,0],r[0,1]], [None, None, r[1,0],r[1,1]])]))
                return slices
            
    def getMask(self):
        self.Mask = self.Image > self.vmin
        
        return self.Mask
        
class polygon:
    def __init__(self, image, dataType='image',xlim=None,ylim=None):
    
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        self.verts = []
    
        # Print instructions for the user
        self.instructions = "\nINSTRUCTIONS TO DRAW THE POLYGON\n\n"+\
           "Select points in the figure by enclosing them within a polygon.\n"+\
           "Press the 'esc' key to start a new polygon.\n"+\
           "Try holding the 'shift' key to move all of the vertices.\n"+\
           "Try holding the 'ctrl' key to move a single vertex."
        
        lineprops =dict(color='r', linestyle='-', linewidth=2, alpha=0.5)
        markerprops = dict(marker='o', markersize=7, mec='r', mfc='r', 
                            alpha=0.5)
        
        # Create the figure and plot the image
        fig = plt.figure(1)
        self.ax1 = fig.add_subplot(111)
        if dataType == 'image':
            self.ax1.imshow(image.T)
        elif dataType == 'points':
            self.ax1.plot(*image.T[0:2],'ob',markersize=3)
            
        if xlim is not None:
            self.ax1.set_xlim(xlim[0],xlim[1])
        if ylim is not None:
            self.ax1.set_ylim(ylim[0],ylim[1])
        
        self.ps = PolygonSelector(self.ax1, self.onselect, lineprops=lineprops,
                                    markerprops=markerprops)
        
        plt.show()


    def onselect(self, verts):
        self.verts = verts[:]

        
    def getPolygon(self):
        return self.verts
        
def crop_image(im,folder=None,fname='rectangle.txt',y_axis=2,x_axis=3,
               return_all=False,scale=1,
               message="Select rectangle to crop the image"):
    if folder is not None: 
        if folder[-1] != "/": folder+="/"
        
    sum_tuple = []
    for i in np.arange(len(im.shape)):
        if i!=y_axis and i!=x_axis: sum_tuple.append(i)
    sum_tuple = tuple(sum_tuple)
    if folder is None:
        rect = rectangle(np.sum(im,axis=sum_tuple),title=message)
        r_c = rect.getRectangle()
    elif not os.path.isfile(folder+fname):
        print(message)
        rect = rectangle(np.sum(im,axis=sum_tuple),title=message)
        r_c = rect.getRectangle()
        np.savetxt(folder+fname,r_c,fmt="%d")
    else:
        r_c = np.loadtxt(folder+fname)
        r_c *= scale
        r_c = r_c.astype(int)
    
    #Crop the image
    im = im.take(indices=range(r_c[0,0],r_c[1,0]), axis=y_axis)
    im = im.take(indices=range(r_c[0,1],r_c[1,1]), axis=x_axis)

    if return_all:
        return im, r_c
    else:
        return im
        
def select_points(points, mask, method='polygon',return_all=False,swap_coords=False):
    if swap_coords: points = points[:,::-1]
    if method=='polygon':
        polygon_mask = geom.polygon.Polygon(mask)
        bool_mask = np.ones(len(points),dtype=bool)
        for n in np.arange(len(points)):
            pt = points[n]
            punto = geom.Point(pt[1],pt[0])
            bool_mask[n] = polygon_mask.contains(punto)
    else:
        bool_mask = (points[:,1] > mask[0,0])* \
                    (points[:,1] < mask[1,0])* \
                    (points[:,0] > mask[0,1])* \
                    (points[:,0] < mask[1,1])
    
    points_out = points[bool_mask]
    if swap_coords: points_out = points_out[:,::-1]
    if return_all:              
        return points_out, bool_mask
    else:
        return points_out
        
def img_line_profile(im,xy=None,return_all=False):
    if xy is None:
        linea = line(im, verbose=False)
        xy = linea.getLine()
    x0, x1 = xy[0,0], xy[1,0]
    y0, y1 = xy[0,1], xy[1,1]
    n_pts = int(np.sqrt((x0-x1)**2 + (y0-y1)**2))
    x, y = np.linspace(x0, x1, n_pts), np.linspace(y0, y1, n_pts)
    zi = scipy.ndimage.map_coordinates(im, np.vstack((x,y)))
    
    if return_all:
        return zi, xy
    else:
        return zi
