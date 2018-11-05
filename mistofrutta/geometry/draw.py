import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import PolygonSelector

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
    
        # Print instructions for the user
        print("\nINSTRUCTIONS TO DRAW THE RECTANGLE\n\n"+\
            "Click and drag the pointer.\n"+\
            "You can resize the rectangle dragging the markers at the "+\
            "corners, and move it dragging the marker at the center.\n"+\
            "You can delete the rectangle clicking anywhere on the image.\n"+\
            "Once you're done, simply close the figure window.")    
            
        # Print additional instructions in case multiple is true
        if self.multiple:
            print("\nMULTIPLE RECTANGLES\n\n"+\
                "You can save multiple rectangles: add the drawn rectangle "+\
                "pressing a and delete the last one pressing d.\n" +\
                "After you add a rectangle, you can proceed directly to draw "+\
                "the next one.\n\n")
        
        rectprops = dict(facecolor='red', edgecolor = 'white',
                 alpha=1.0, fill=False, lw=2)
        
        # Create the figure and plot the image
        fig = plt.figure(1)
        self.ax1 = fig.add_subplot(111)
        self.ax1.imshow(image.T)
        
        self.rs = RectangleSelector(self.ax1, self.line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True, rectprops=rectprops)
                       
        plt.connect('key_press_event', self.keyboardCallback)
        
        plt.show()


    def line_select_callback(self,eclick,erelease):
        return 1
        
    
    def keyboardCallback(self,event):
        if event.key in ['A', 'a']:
            if self.multiple:
                self.x1 = np.min(self.rs.corners[0])
                self.x2 = np.max(self.rs.corners[0])
                self.y1 = np.min(self.rs.corners[1])
                self.y2 = np.max(self.rs.corners[1])
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
                

        
    def getRectangle(self):
        # Returns a single rectangle if self.multiple is false.
        # Otherwise, it returns an array of rectangles.
        if not self.multiple:
            self.x1 = np.min(self.rs.corners[0])
            self.x2 = np.max(self.rs.corners[0])
            self.y1 = np.min(self.rs.corners[1])
            self.y2 = np.max(self.rs.corners[1])
            
            return np.array([[self.x1,self.y1],[self.x2,self.y2]])
        else:
            return np.array(self.rectangles)
        
class polygon:
    def __init__(self, image, dataType='image'):
    
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        self.verts = []
    
        # Print instructions for the user
        print("\nINSTRUCTIONS TO DRAW THE POLYGON\n\n"+\
           "Select points in the figure by enclosing them within a polygon.\n"+
           "Press the 'esc' key to start a new polygon.\n"+
           "Try holding the 'shift' key to move all of the vertices.\n"+
           "Try holding the 'ctrl' key to move a single vertex.")
        
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
