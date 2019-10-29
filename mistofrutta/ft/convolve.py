import numpy as np
from mistofrutta.ft import ft_cubic

def convolve(A,B,X):
    a = X[0]
    b = X[-1]
    delta = X[1]-X[0]
    
    nyquist = np.pi/delta
    a_o = -nyquist
    b_o = nyquist
    delta_o = (b_o-a_o)/(X.shape[0])/10
    Omega = np.arange(a_o,b_o,delta_o)
    
    FA = ft_cubic(A,a,delta,Omega)
    FB = ft_cubic(B,a,delta,Omega)
    FAB = FA*FB
    
    AB = np.absolute(ft_cubic(FAB,a_o,delta_o,-X))
    
    return AB
