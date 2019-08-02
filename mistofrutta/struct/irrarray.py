import numpy as np

class irrarray(np.ndarray):
    '''
    Extension of numpy ndarray that supports "irregular" strides. 
    
    This serves mostly just as a shorthand notation for slices of the type
    A[B[i]:B[i+1]], where B contains the first indices of contiguous blocks of
    elements of A (the slice returning "block" i). Using this class, B can be 
    associated with A at instantiation with A = irrarr(A, B), so that block i
    can be now retrieved with A(i).
    
    Multiple irregular stridings can exist at once, with names specified via
    strideNames. Block i of type p can be retrieved with A(p=i), and
    A(p=i,q=j) returns a list equivalent to [A(p=i),A(q=j)].
    
    This class was originally written to deal with sets of points in 3D space,
    unevenly distributed in different blocks (volumes). There are some names 
    that are reserved and when passed as argument to A(name=i) make the call 
    behave differently than when using strideNames. By default, these names are 
    z,y,x. A(z=i) returns A[np.where(A[:,column]==i], where column is 0,1,2 for 
    z,y,x respectively. Again, this is just a shorthand notation for it.
    You can change the column reserved names via the parameter columnNames.
    '''

    def __new__(cls, input_array, irrStrides, strideNames=["k"], columnNames=["z","y","x"]):
        obj = np.asarray(input_array).view(cls)
        obj.irrStrides = irrStrides
        obj.columnNames = columnNames
        
        if type(irrStrides)!=list: irrStrides=[irrStrides]
                
        if not len(irrStrides)==len(strideNames): print("error")
        obj.upToIndex = {}
        
        for i in np.arange(len(irrStrides)):
            name = strideNames[i]
            if name in obj.columnNames:
                print("The strideName "+name+" conflicts with one of the \
                       columnNames. Dropping the stride "+name+". Note, by \
                       default the names z, y, and x are reserved and assigned \
                       to the columns 0, 1, and 2, respectively. To avoid this \
                       pass an empty list as columnNames.")
            else:
                obj.upToIndex[name] = np.zeros(len(irrStrides[i])+1,dtype=int)
                obj.upToIndex[name][1:] = np.cumsum(irrStrides[i])
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.columnNames = getattr(obj, 'columnNames', ["z","y","x"])
        self.upToIndex = getattr(obj, 'upToIndex', {})
        
    def __array_wrap__(self, out_arr, context=None):
        return repr(out_arr)
        
    def __call__(self, k=None, **kwargs):
        if k!=None:
            i0 = self.upToIndex["k"][k]
            i1 = self.upToIndex["k"][k+1]
            return self[i0:i1]
        
        tbReturned = []
        for key in kwargs:
            if key in self.columnNames:
                columnIndex = self.columnNames.index(key)
                k = kwargs[key]
                tbReturned.append(self[np.where(self[:,columnIndex]==k)])
            else:
                k = kwargs[key]
                i0 = self.upToIndex[key][k]
                i1 = self.upToIndex[key][k+1]
                tbReturned.append(self[i0:i1])
        
        if len(tbReturned)==1: tbReturned=tbReturned[0]
        return tbReturned
