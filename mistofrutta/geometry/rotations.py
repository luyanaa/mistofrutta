import numpy as np
from scipy.interpolate import RegularGridInterpolator
import mistofrutta as mf

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
    
def rotate_hyperstack(im, ch_insp=0, ch_axis=1, return_all=False):
    A = np.max(im.take(indices=ch_insp, axis=ch_axis),axis=0)
    # Draw the line that should become horizontal
    instr = "CENTERLINE: Click two points defining the centerline. First posterior, then anterior."
    linea = mf.geometry.draw.line(A,verbose=True,custom_instructions=instr)
    ll = linea.get_line()
    tan = (ll[1,1]-ll[0,1])/(ll[1,0]-ll[0,0])
    theta = np.arctan(tan)
    # Select the origin for the rotation
    instr = "ORIGIN: Click one point at the center of rotation."
    centro = mf.geometry.draw.line(A,verbose=True,custom_instructions=instr)
    cc = centro.get_line()
    x0 = cc[0,1]
    y0 = cc[0,0]
    
    #Do the rotation - purely around z
    rot_im = np.zeros_like(im)
    for ch_j in np.arange(im.shape[ch_axis]):
        tmp = im.take(indices=ch_j, axis=ch_axis)
        idx=[slice(None)]*rot_im.ndim
        idx[ch_axis] = ch_j
        rot_im[tuple(idx)] = rotate_3D_image(tmp, theta-np.pi/2., 1.,0.,0.,0.,y0,x0)
    
    if return_all:
        return rot_im,theta-np.pi/2.,1.,0.,0.,0.,y0,x0
    else:
        return rot_im
        
    
def rotate_3D_image(A, theta, uz, uy, ux, z0=0.0, y0=0.0, x0=0.0):
    '''
    Rotates a 3D image along the axis (uz,uy,ux), optionally with respect to a
    new specified origin of the axes.
    
    Parameters
    ----------
    A: numpy array
        3D array containing the image to be rotated
    theta: float
        angle of the rotation
    uz, uy, ux: float
        components of the vector defining the axis of rotation
    z0, y0, x0: float
        optional coordinates of the new origin of the axes.
        
    Returns
    -------
    Aprime: Rotated image
    '''
    
    nz = A.shape[0]
    ny = A.shape[1]
    nx = A.shape[2]
    
    Z = np.arange(nz)-z0
    Y = np.arange(ny)-y0
    X = np.arange(nx)-x0
    
    ZZ, YY, XX = np.array(np.meshgrid(Z,Y,X,indexing='ij'))
    ZZ = ZZ.reshape(nz*ny*nx)
    YY = YY.reshape(nz*ny*nx)
    XX = XX.reshape(nz*ny*nx)
    
    ZYX = np.array([ZZ,YY,XX])
    
    interpolating_function = RegularGridInterpolator((Z,Y,X), A, 
                                            bounds_error=False, fill_value=0.0)
    
    ZYXprime = np.dot(matrix(theta, uz, uy, ux),ZYX)
    
    Aprime = interpolating_function(ZYXprime.T)
    Aprime = Aprime.reshape((nz,ny,nx))

    return Aprime
