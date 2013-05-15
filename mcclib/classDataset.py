import numpy as np
import os
import constants
import utils
from utils import info
import cPickle

ending = ".npy"

class ArrayFile(object):
    def __init__(self, name, dim, dtype, filename=None, fileprefix="A", onDisk=False):
        self.name = name
        self.dim = dim
        self.dtype = dtype
        self.fileprefix = fileprefix
        self.onDisk = onDisk
    
    def getFilename(self):
        if self.fileprefix is not None:
            filename = self.fileprefix + "_" + self.name + ending
        else:
            filename = self.name + ending
        return filename

class Dataset(object):
    
    is_amoeboid = 0
    is_mesenchymal = 1
    
    AMOEBOID = {"times" : ArrayFile("times", "NNN", np.float_),
                "types" : ArrayFile("types", "N_agents", np.int_),
              "positions" : ArrayFile("positions", "shapexDim", np.float_, onDisk=True),
              "velocities" : ArrayFile("velocities", "shapexDim", np.float_, onDisk=True),
              "energies" : ArrayFile("energies", "shapex1", np.float_),
              "states" : ArrayFile("states", "shapex1", np.int8),
              "statechanges" : ArrayFile("statechanges", "N_agents", np.float_),
              "periods" : ArrayFile("periods", "N_agents", np.float_),
              "delays" : ArrayFile("delays", "N_agents", np.float_),
              "eating" : ArrayFile("eating", "shapex1", np.bool_),
              "direction_angles" : ArrayFile("direction_angles", "N_agents", np.float_)}

    def __init__(self, arrays, max_time, dt, N_agents, N_dim, path, doAllocate=True, fileprefix=None):
        '''
        Constructor
        '''
        self.dt = dt
        self.arrays = arrays
        self.NNN = int(max_time / dt)
        self.shapexDim = (self.NNN, N_agents, N_dim)
        self.shapex1 = (self.NNN, N_agents)
        self.shapexEdim = (self.NNN, constants.E_dim, N_agents)
        self.N_agents = N_agents
        self.fileprefix = fileprefix
        self.path = path
        
        if doAllocate:
            for name, val in self.arrays.iteritems():
                if val.onDisk:
                    filename = os.path.join(self.path, val.getFilename())
                    arr = np.memmap(filename, dtype=val.dtype, mode='w+', shape=self.__getattribute__(val.dim))
                    setattr(self, name, arr)
                else:
                    setattr(self, name, np.zeros(self.__getattribute__(val.dim), dtype=val.dtype))
        else:
            for name, val in self.arrays.iteritems():
                setattr(self, name, np.empty(self.__getattribute__(val.dim), dtype=val.dtype))
    
    def getTotalSize(self):
        return self.getSize(False)+self.getSize(True)
    
    def getSize(self, onDisk):
        totalbytes = 0
        for name, val in self.arrays.iteritems():
            if val.onDisk==False:
                a = self.__getattribute__(name)
                totalbytes += a.size * a.itemsize
        return totalbytes
                
    def getHumanReadableSize(self):
        size = self.getTotalSize()
        bytesInMB = 1024*1024
        return "%s MB" % int(size/bytesInMB) 
    
    def resizeTo(self, iii):
        if iii==self.NNN:
            return
        self.NNN = iii
        self.shapexDim = (iii, self.shapexDim[1], self.shapexDim[2])
        self.shapex1 = (iii, self.shapex1[1])
        
        for name, val in self.arrays.iteritems():
            self.__getattribute__(name).resize(self.__getattribute__(val.dim), refcheck=False)
    
    def saveTo(self, path):
        for name, val in self.arrays.iteritems():
            arr = self.__getattribute__(name)
            if type(arr) == np.memmap:
                filename = os.path.join(path, val.getFilename() + ".shape.pickle")
                with open(filename, "w") as picklefile:
                    cPickle.dump(arr.shape, picklefile)
                info("Memmap %s saved" % name)
                del arr
            else:
                filepath = os.path.join(path, val.getFilename())
                np.save(filepath, arr)
    
    def erase(self):
        mmapfiles = []
        for name in self.arrays:
            arr = self.__getattribute__(name)
            if type(arr) == np.memmap:
                mmapfiles.append(arr.filename)
                del arr
        for delfile in mmapfiles:
            utils.deleteFile(delfile)
    
#    def release(self):
#        for name in self.arrays:
#            self.__delattr__(name)

def load(arrays, path, fileprefix, dt, readOnly=True):
    """Loads a *dataset* with the specified *array* configuration from path, using a certain *fileprefix*."""
    try:
        loaded_arrays = dict()
        for name, val in arrays.iteritems():
            filepath = os.path.join(path, val.getFilename())
            if val.onDisk==False:
                loaded_arrays[name] = np.load(filepath)
            else:
                with open(filepath + ".shape.pickle") as picklefile:
                    arrshape = cPickle.load(picklefile)
                    arrmode = "r" if readOnly else "w+"
                    arr = np.memmap(filepath, dtype=val.dtype, mode=arrmode, shape=arrshape)
                    loaded_arrays[name] = arr
    
        sample_arr = loaded_arrays["energies"]
        
        ds = Dataset(arrays, sample_arr.shape[0], dt, sample_arr.shape[1], constants.DIM, path, doAllocate=False, fileprefix=fileprefix)
        for name, val in arrays.iteritems():
            ds.__setattr__(name, loaded_arrays[name])
    except IOError, ex:
        print ex
        return None
    return ds