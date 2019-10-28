import numpy as np
from mistofrutta.ft import ft_cubic

def convolve(A,B,X):
    a = X[0]
    delta = X[1]-X[0]

    nyquist = np.pi/delta
    a_o = -nyquist
    b_o = nyquist
    delta_o = (b_o-a_o)/(X.shape[0]-1)
    Omega = np.arange(a_o,b_o,delta_o)
    
    FA = ft_cubic(A,a,delta,Omega)
    FB = ft_cubic(B,a,delta,Omega)
    FAB = FA*FB
    AB = ft_cubic(FAB,a_o,delta_o,-X)
    '''
    Freq = np.fft.fftfreq(X.shape[0],delta)
    FA = np.fft.fft(A)*np.exp(-2.j*np.pi*Freq*a)
    FB = np.fft.fft(B)*np.exp(-2.j*np.pi*Freq*a)
    FAB = FA*FB
    AB = np.fft.ifft(FAB)'''
    
    return AB
