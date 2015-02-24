import numpy as np
import matplotlib.pyplot as plt

def height_avg(grid, arr, j_bot, j_top):
    assert(j_bot < j_top)
    dz = grid.dz[j_bot:j_top]
    integral = (arr[:, j_bot:j_top] * dz).sum(axis=1) / dz.sum()
    return integral

def r_mag(grid, df, omega_star):
    """
    Computes the magnetospheric radius using 3 separate metrics.
    """
    j_bot, j_top = grid.nztot/2-30, grid.nztot/2+30+1

    # Metric 1: use the location where the plasma \beta = 1
    beta_avg = height_avg(grid, df.beta, j_bot, j_top)
    i_beta = 0
    while (beta_avg[i_beta] < 1.0): i_beta += 1
    
    # Metric 2: use the location where the kinetic plasma \beta_1 = 1
    beta1_avg = height_avg(grid, df.beta1, j_bot, j_top)
    i_beta1 = 0
    while (beta1_avg[i_beta1] < 1.0): i_beta1 += 1

    # Metric: identify last location in the disk where \Omega = \Omega_*
    j_bot, j_top = grid.nztot/2-3, grid.nztot/2+3+1
    omega_avg = height_avg(grid, df.omega, j_bot, j_top)
    i_omega = grid.nrtot-1
    while((omega_avg[i_omega] < omega_star) and (i_omega>0)): i_omega -= 1;
    return i_beta, i_beta1, i_omega

def mdot(grid, df):
    rdz = (grid.r * grid.dz[:, np.newaxis]).transpose()
    rdr = (grid.r * grid.dr)[:, np.newaxis] * np.ones(grid.shape)
    rho, vr, vz = df.rho, df.vr, df.vz

    mdot_r = -2*np.pi* rho * vr * rdz 
    mdot_z = 2*np.pi* rho * vz * rdr * np.sign(-grid.z[np.newaxis, :])
    return mdot_r, mdot_z

def find_index(grid, dist):
    """
    Identifies indices of grid corresponding to a cylinder with 
    height 2*dist and radius dist.
    """
    r_index = np.where(grid.r >= dist)[0][0]
    z_topindex = np.where(grid.z >= dist)[0][0]
    z_botindex = np.where(grid.z <= -dist)[0][-1]
    return r_index, z_botindex, z_topindex

def mdot_disk(grid, df, dist):
    """
    Computes radial mass flux through the disk
    """
    i, _, _ = find_index(grid, dist)
    mdot_r, _ = mdot(grid, df) 
    mdot_r_in, mdot_r_out = 0.0, 0.0
    
    for j in range(grid.nztot/2-61, grid.nztot/2+60):
        temp = mdot_r[i, j]
        if temp > 0:
            mdot_r_in += temp
        else:
            mdot_r_out += temp

    return mdot_r_in, mdot_r_out

def mdot_through_surface(grid, df, dist):
    """
    Computes the mass flux through a cylindrical surface 
    with height 2*dist and radius dist centered on the origin.
    """
    i, j_bot, j_top = find_index(grid, dist)
    assert j_top > j_bot
    mdot_r, mdot_z = mdot(grid, df)        
    return mdot_r[i, j_bot:j_top].sum() + \
        mdot_z[:i, j_top].sum() + \
        mdot_z[:i, j_bot].sum()

