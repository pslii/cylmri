
class SimData:
    """
    Wrapper for output of 2D MRI simulations.
    """
    
    def __init__(self, grid, data_dict, metadata_dict):
        self.shape = grid.shape
        fluidvars = ['rho', 'p', 's', 
                     'hr', 'hf', 'hz', 
                     'u', 'v', 'w', 'psi']
                
        (self.rho, self.p, self.s,
         self.hr, self.hf, self.hz,
         self.vr, self.vphi, self.vz, self.psi) = \
                                                  [data_dict.get(var) for var in fluidvars]
        self.temp = self.p / self.rho
        self.data = data_dict
        self.metadata = metadata_dict
            
    def keys(self):
        return self.data.keys() + self.metadata.keys()
