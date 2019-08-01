import numpy as np

class irrarray(np.ndarray):
    '''
    Extension of numpy ndarray that supports "irregular" strides. In the end, it
    is just a way to make slices more readable in the following case. Say that
    you have an array A, and contiguous slices of A semantically belong to 
    different blocks, with the numbers of elements belonging to each block being
    uneven, i.e. you cannot just reshape A. You could use a separate "reference"
    array B containing the indices of the first element of each block, but then
    to obtain block i you would do A[B[i]:B[i+1]], with possibly A and B being
    longer names. This class allows you to hide B and use A(i) to obtain the 
    slice corresponding to the i-th block. Compared to splitting the blocks and
    having A as a list of numpy arrays, using this class has the advantage that
    A[j] still behaves as the original numpy array in all the numpy functions,
    slicing, etc.
    Also supports multiple irregular stridings that you can name, such that you
    can call A(v=1), A(f=2), or even A(v=1,f=2).
    '''

    def __new__(cls, input_array, irrStrides, strideNames=["k"]):
        obj = np.asarray(input_array).view(cls)
        obj.irrStrides = irrStrides
        
        if type(irrStrides)!=list: irrStrides=[irrStrides]
                
        if not len(irrStrides)==len(strideNames): print("error")
        obj.upToIndex = {}
        
        for i in np.arange(len(irrStrides)):
            name = strideNames[i]
            obj.upToIndex[name] = np.zeros(len(irrStrides[i])+1,dtype=int)
            obj.upToIndex[name][1:] = np.cumsum(irrStrides[i])
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        
    def __array_wrap__(self, out_arr, context=None):
        return repr(out_arr)
        
    def __call__(self, k=None, **kwargs):
        if k!=None:
            i0 = self.upToIndex["k"][k]
            i1 = self.upToIndex["k"][k+1]
            return self[i0:i1]
        
        tbReturned = []   
        for key in kwargs:
            k = kwargs[key]
            i0 = self.upToIndex[key][k]
            i1 = self.upToIndex[key][k+1]
            tbReturned.append(self[i0:i1])
            return tbReturned
