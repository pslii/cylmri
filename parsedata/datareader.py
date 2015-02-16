import numpy as np
import struct
from simdata import SimData
import time

class DataReader:

    VARLIST_2DMRI = ['rho', 'p', 's', 
                    'hr', 'hf', 'hz', 
                     'u', 'v', 'w',
                     'psi', 'rhoadd', 
                     'time', 'ko', 'it', 'tau', 'zdisk']
    FMTLIST_2DMRI = ['darr', 'darr', 'darr',
                     'darr', 'darr', 'darr',
                     'darr', 'darr', 'darr',
                     'darr', 'darr', 
                     'd', 'i', 'i', 'd', 'd']

    def __init__(self, grid, 
                 fmtlist=FMTLIST_2DMRI, 
                 varlist=VARLIST_2DMRI):
        self.grid = grid
        self.nr = grid.nrtot/grid.nlayers_i # size of each individual data block
        self.nz = grid.nztot/grid.nlayers_j
        self.nblocks = grid.nlayers_i * grid.nlayers_j
        self.block_shape = (self.nblocks, self.nz + 4, self.nr + 4)
        self.block_size = np.prod(self.block_shape)
        
        assert(len(fmtlist) == len(varlist))
        self.unpacklist, self.fmtlist, self.varlist, self.filesize = \
                                                                     self._unpackdata(fmtlist, varlist)

    @staticmethod
    def getByteSize(c):
        assert (len(c) == 1)
        if c in 'd':
            return 8
        elif c in 'fil':
            return 4
        elif c in 'c?':
            return 1
        else:
            print "Error: invalid character format"
            assert False

    def _unpackdata(self, fmtlist, varlist):
        """
        Computes start and end indices of arrays/values in the binary file.
        """
        fmtlist_out, varlist_out = [], []
        formats, starts, ends = [], [], []
        start, end = 0, 0

        for fmt, var, i in zip(fmtlist, varlist, range(len(fmtlist))):
            start = end
            block_size = self.block_size if ('arr' == fmt[1:]) else 1
            end += block_size * self.getByteSize(fmt[0])

            if not (var == '' or var == '_'):  # strip out unused arrays
                fmtlist_out.append(fmt)
                varlist_out.append(var)
                formats.append('<' + fmt[0] * block_size)
                starts.append(start)
                ends.append(end)
        sizes = np.array(ends) - np.array(starts)
        return zip(starts, sizes, formats), \
            fmtlist_out, \
            varlist_out, \
            sizes.sum()

    def _repack(self, fmt, binary):
        unpack = struct.unpack(fmt, binary)
        if len(unpack) > 1:
            var = np.empty(self.grid.shape)
            unpack = np.reshape(unpack, self.block_shape)
            for l in range(self.nblocks):
                li = l % self.grid.nlayers_i
                lj = l / self.grid.nlayers_i
                var[li * self.nr:(li + 1) * self.nr, \
                    lj * self.nz:(lj + 1) * self.nz] = unpack[l, \
                                                              2:self.nz+2, \
                                                              2:self.nr+2].transpose()

        else:
            var = unpack
        return var

    def _progressbar(self, fname, progress, total):
        """
        Outputs a progressbar.
        Reading 0039dat: [==================================================] 100%

        Note: need to set -u flag in python call for this to work properly
        :param progress:
        :param total:
        :return:
        """
        percentage = int(progress * 100.0 / total)
        percentage_bar = ('=' * (percentage / 2)).ljust(50)
        print '\rReading {0}: [{1}] {2}%'.format(fname, percentage_bar, percentage),

    @staticmethod
    def filename(ndat, suffix='dat', n_digits=4):
        formatstr = "{0:0"+str(n_digits)+"d}"+suffix
        return formatstr.format(ndat)
   
    def filenames(self, suffix='dat', n_digits=4, path='.'):
        """
        Generator for datafile names.
        """

    def files(self, suffix="dat", n_digits=4, path='.'):
        import os.path
        fnum = 0
        formatstr = "{0:0"+str(n_digits)+"d}"+suffix
        filename = formatstr.format(fnum)
        while(os.path.isfile(path+'/'+filename)
              and fnum <= (10**n_digits-1)):
            yield self.readData(filename, path=path)
            fnum += 1
            filename = formatstr.format(fnum)
        
    def readData(self, fname, path='.'):
        t0 = time.time()
        data, metadata = {}, {}
        bytesRead = 0
        self._progressbar(fname, bytesRead, self.filesize)

        with open(path + '/' + fname, 'rb') as datfile:
            for ((start, size, fmt), varname) in zip(self.unpacklist, self.varlist):
                datfile.seek(start)
                binary = datfile.read(size)
                var = self._repack(fmt, binary)

                if isinstance(var, np.ndarray) and len(var) > 1:
                    assert var.shape == self.grid.shape
                    data[varname] = var
                else:
                    metadata[varname] = var[0]
                bytesRead += size
                self._progressbar(fname, bytesRead, self.filesize)
        t1 = time.time()
        print " {:.2f} s".format(t1-t0)

        return SimData(self.grid, data, metadata)
        

