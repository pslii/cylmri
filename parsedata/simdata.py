import numpy as np

class SimData:
    """
    Wrapper for output of 2D MRI simulations.
    """
    debug = True
    
    def __init__(self, params, grid, data_dict, metadata_dict):
        self.shape = grid.shape
        fluidvars = ['rho', 'p', 's', 
                     'hr', 'hf', 'hz', 
                     'u', 'v', 'w', 'psi']
                
        # primary fluid/magnetic variables
        (self.rho, self.p, self.s,
         self.hr_raw, self.hf, self.hz_raw,
         self.vr, self.vphi, self.vz, self.psi) = \
                                                  [data_dict.get(var) for var in fluidvars]
        self.hr_d, self.hz_d = self.compute_dipole(grid, params['dip'])
        self.hr = self.hr_raw + self.hr_d
        self.hz = self.hz_raw + self.hz_d
                
        # secondary variables
        self.omega = self.vphi / grid.r[:,np.newaxis]
        if self.debug:
            assert(np.all(self.vphi[:,grid.nztot/2] / grid.r == self.omega[:,grid.nztot/2]))
            assert(np.all(self.vphi[:,-1] / grid.r == self.omega[:,-1]))
            assert(np.all(self.vphi[:,0] / grid.r == self.omega[:,0]))

        self.beta, self.beta1 = self._compute_beta()

        self.temp = self.p / self.rho
        self.grid = grid
        self.data = data_dict
        self.metadata = metadata_dict
            
    def _compute_beta(self):
        p_mag = (self.hr**2 + self.hf**2 + self.hz**2)/(8 * np.pi)
        v2 = self.vr**2 + self.vphi**2 + self.vz**2
        p_ram = self.rho*v2
        beta = self.p / p_mag
        beta1 = (p_ram + self.p) / p_mag
        return beta, beta1

    def compute_dipole(self, grid, dip_strength):
        r_sph = (grid.r2D**2 + grid.z2D**2)**0.5
        rz = grid.r[:, np.newaxis]*grid.z

        if self.debug:
            assert(np.all(grid.r * grid.z[grid.nztot/2] == rz[:,grid.nztot/2]))
            assert(np.all(grid.r * grid.z[0] == rz[:,0]))
            assert(np.all(grid.r * grid.z[-1] == rz[:,-1]))

        hr_dip = 3.0*dip_strength*rz / r_sph**5
        hz_dip = (3.0*dip_strength*grid.z2D**2 / r_sph**5) - \
                 (dip_strength / r_sph**3)
        return hr_dip, hz_dip

    def keys(self):
        return self.data.keys() + self.metadata.keys()

