import numpy as np
from scipy.interpolate import RegularGridInterpolator

def matrix(theta, ux, uy, uz):
    '''
    Returns the transformation matrix for a rotation in 3D of an angle theta 
    around the axis defined by the vector (ux, uy, uz).
    
    Parameters
    ----------
    theta: float
        angle of the rotation
    ux, uy, uz: float
        components of the vector defining the axis of rotation
        
    Returns
    -------
    R: 3x3 matrix describing the transformation
    
    '''
    norm = np.sqrt(ux**2 + uy**2 + uz**2)
    ux /= norm
    uy /= norm
    uz /= norm
    
    R = np.zeros((3,3), dtype=np.float64)
    costheta = np.cos(theta)
    oneminuscostheta = 1. - costheta
    sintheta = np.sin(theta)
    
    R[0,0] = costheta + ux**2 * (oneminuscostheta)
    R[0,1] = ux*uy*(oneminuscostheta) - uz * sintheta
    R[0,2] = ux*uz*(oneminuscostheta) + uy * sintheta
    
    R[1,0] = ux*uy*(oneminuscostheta) + uz * sintheta
    R[1,1] = costheta + uy**2 * (oneminuscostheta)
    R[1,2] = uy*uz * (oneminuscostheta) - ux * sintheta
    
    R[2,0] = uz*ux*(oneminuscostheta) - uy*sintheta
    R[2,1] = uz*uy*(oneminuscostheta) + ux*sintheta
    R[2,2] = costheta + uz**2 * (oneminuscostheta)
    
    return R
    
def rotate_3D_image(A, theta, ux, uy, uz, x0=0.0, y0=0.0, z0=0.0):
    '''
    Rotates a 3D image along the axis (ux,uy,uz), optionally with respect to a
    new specified origin of the axes.
    
    Parameters
    ----------
    A: numpy array
        3D array containing the image to be rotated
    theta: float
        angle of the rotation
    ux, uy, uz: float
        components of the vector defining the axis of rotation
    x0, y0, z0: float
        optional coordinates of the new origin of the axes.
        
    Returns
    -------
    Aprime: Rotated image
    '''
    
    nx = A.shape[0]
    ny = A.shape[1]
    nz = A.shape[2]
    
    X = np.arange(nx)-x0
    Y = np.arange(ny)-y0
    Z = np.arange(nz)-z0
    
    interpolating_function = RegularGridInterpolator((X,Y,Z), A, 
                                            bounds_error=False, fill_value=-1.0)
                                            
    XYZ = np.array([X,Y,Z])
    XYZprime = matrix(theta, ux, uy, uz).dot(XYZ)
    
    GridPrime = np.array(np.meshgrid(XYZprime[0],XYZprime[1],XYZprime[2])).T.reshape((nx*ny*nz,3))

    Aprime = interpolating_function(GridPrime)

    Aprime = Aprime.reshape((nx,ny,nz))

    return Aprime
