import numpy as np
import struct

class GridReader:
    """
    Reads in and parses bindary grid file for 2.5D cylindrical coordinates.
    """
    debug = True
    
    def __init__(self, params, fname='grid', path='.'):
        self.nrtot = params['nxtot']
        self.nztot = params['nytot']
        self.shape = (self.nrtot, self.nztot)
        self.nlayers_i = params['nlayers_radius']
        self.nlayers_j = params['nlayers_angle']
        
        self.r_edge, self.z_edge = self._readgrid(fname, path)
        self.r, self.z = self._cellcoords()
        self.dr, self.dz = self._cellwidth()        
        self.r2D, self.z2D = self._twod(self.r, self.z)
        self.dr2D, self.dz2D = self._twod(self.dr, self.dz)

    def _readgrid(self, fname, path):
        """
        Reads in and parses the raw binary grid information.
        """
        with open(path + '/' + fname, 'rb') as gridfile:
            binary = gridfile.read()
            start, num = 0, self.nrtot + 5
            r_edge = np.array(struct.unpack('<' + 'd' * num,
                                            binary[:num * 8]))
            start, num = self.nrtot + 5, self.nztot + 5
            z_edge = np.array(struct.unpack('<' + 'd' * num,
                                            binary[start * 8:(start + num) * 8]))
        return r_edge, z_edge
            

    def _cellcoords(self):
        """
        Computes the coordinates of the cell centers from the edge coordinates.
        """
        r_edge, z_edge = self.r_edge, self.z_edge
        r = (np.roll(r_edge, -1) + r_edge)[2:-3] / 2.0
        z = (np.roll(z_edge, -1) + z_edge)[2:-3] / 2.0
        return r, z

    def _cellwidth(self):
        """
        Computes the widths of each cell.
        """
        dr = (np.roll(self.r_edge, -1) - self.r_edge)[2:-3]
        dz = (np.roll(self.z_edge, -1) - self.z_edge)[2:-3]
        return dr, dz

    def _twod(self, r, z):
        r2D = r[:,np.newaxis] * np.ones(self.nztot)
        z2D = np.ones((self.nrtot,1)) * z
        if self.debug:
            assert(np.all(r == r2D[:,0]))
            assert(np.all(r == r2D[:,self.nztot/2]))
            assert(np.all(r == r2D[:,-1]))

            assert(np.all(z == z2D[0,:]))
            assert(np.all(z == z2D[self.nrtot/2,:]))
            assert(np.all(z == z2D[-1,:]))
        return r2D, z2D

    def broadcast(self, r, z):
        return r * z[:, np.newaxis]

