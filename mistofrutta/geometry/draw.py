import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import PolygonSelector
from matplotlib.widgets import Slider

class line:
    def __init__(self, image):
    
        self.X = []
        self.Y = []
    
        # Print instructions for the user
        print("\nINSTRUCTIONS TO DRAW THE LINE\n\n"+\
            "Click to add a point. Right-click to delete the last one. \n"+\
            "***Add one point outside the image in the posterior direction."+\
            "***\n"+\
            "Ignore the first point in the corner of the image. \n" +\
            "Once you're done clicking, simply close the figure window.")    
        
        # Create the figure and plot the image
        fig = plt.figure(1)
        ax1 = fig.add_subplot(111)
        ax1.imshow(image.T)
        self.line, = ax1.plot(0,0,'*k')
        
        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)
        
        # Start displaying the plot
        plt.show()

    # Define callback function to be called when there is a mouse click
    def __call__(self, event):
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
        return np.array([self.X,self.Y]).T #[::-1]
        
        
class rectangle:
    def __init__(self, image, multiple=False):
    
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        
        self.multiple = multiple
        self.rectangles = []
        
        self.Image = image
    
        # Print instructions for the user
        if self.multiple: print("Multiple rectangles. ")
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
        
        # Set a custom viridis color map useful for thresholding
        self.cmap = plt.cm.viridis
        cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        cmaplist[0] = (1.,0.,0.,1.)
        cmaplist[-1] = (1.,0.5,0.,1.)
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, self.cmap.N)
        
        # Plot the image
        self.ax1.imshow(self.Image.T,cmap=self.cmap)
        
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
    def __init__(self, image, dataType='image'):
    
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
        else:
            self.ax1.plot(*image.T[0:2],'ob',markersize=3)
            self.ax1.set_xlim(0,512)
            self.ax1.set_xlim(0,512)
        
        self.ps = PolygonSelector(self.ax1, self.onselect, lineprops=lineprops,
                                    markerprops=markerprops)
        
        plt.show()


    def onselect(self, verts):
        self.verts = verts[:]

        
    def getPolygon(self):
        return self.verts
