import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from time import strftime, gmtime

r_sun = 6.9570000000000000E+010
r_tot = 5.7671763407229273 * r_sun
r_env_init = 0.2836686568328816 * r_tot

struc_dir = "strucdir/*"

all_files = sorted(glob.glob(struc_dir))

def equal_spaced_indices(lst, n):
    """Get n equally spaced indices from first to last"""
    indices = np.linspace(0, len(lst)-1, n, dtype=int)
    return [lst[i] for i in indices]

# sub_files = equal_spaced_indices(all_files, 2)
# sub_files = all_files
sub_files = [all_files[-1]]

headers = ['index', 'radius', 'vel_rms', 'crossing_time_vel', 'crossing_time_hp', 'press_height', 'e_int_spec', 'nabla', 'nabla_adiab', 'pressure', 'temperature', 'entropy', 'entropy_calc', 'e_kin', 'shell_mass', 'density', 'flux_enthalpy', 'flux_e_kin', 'flux_rad', 'flux_diff_scalar', 'vel_1_rms', 'vel_2_rms', 'vel_3_rms', 'vel_1', 'vel_2', 'vel_3', 'theta_1', 'u_phi_average_1', 'theta_2', 'u_phi_average_2', 'theta_3', 'u_phi_average_3', 'theta_4', 'u_phi_average_4', 'theta_5', 'u_phi_average_5', 'theta_6', 'u_phi_average_6', 'theta_7', 'u_phi_average_7', 'theta_8', 'u_phi_average_8', 'theta_9', 'u_phi_average_9', 'theta_10', 'u_phi_average_10', 'theta_11', 'u_phi_average_11', 'angular_momentum', 'scalar', 'flux_mass', 'nabla_mu_equiv', 'flux_heat_instant', 'flux_heat_init', 'flux_heat_ref_prof', 'flux_heat_ref_field', 'brunt_vaissala', 'temperature_std', 'temperature_skew', 'temperature_kurt', 'temperature_rms', 'density_std', 'density_skew', 'density_kurt', 'density_rms', 'temperature_fluct', 'temperature_up_fluct', 'temperature_down_fluct', 'vel_1_correlation', 'buoyancy_work', 'vel_1_std', 'vel_1_skew', 'vel_1_kurt', 'vel_1_rms', 'temperature_gradient', 'thermal_conductivity', 'strain_dev_1_1_squared', 'strain_dev_2_2_squared', 'strain_dev_3_3_squared', 'strain_dev_1_2_squared', 'strain_dev_1_3_squared', 'strain_dev_2_3_squared']

def read_file(fname):
    with open(fname, "r") as f:
        f.readline()
        time = f.readline().split()[0]
    df = pd.DataFrame(np.genfromtxt(fname,skip_header=3),columns=headers)
    df['time'] = time
    return df

fig, ax = plt.subplots()
ax.set_xlabel("radius")
ax.set_ylabel(r"$v_\mathrm{1,rms}$")


fig1, ax1 = plt.subplots()
ax1.set_xlabel("radius")
ax1.set_ylabel(r"$v_\mathrm{2,rms}$")

fig2, ax2 = plt.subplots()
ax2.set_xlabel("radius")
ax2.set_ylabel(r"$v_\mathrm{3,rms}$")

for i in sub_files:
    df = read_file(i)
    ax.semilogy(df['radius'] / r_tot, df['vel_1_rms'], label=df['time'][0])
    ax.set_title(i)
    ax1.plot(df['radius'] / r_tot, df['vel_2_rms'], label=df['time'][0])
    ax1.set_title(i)
    ax2.plot(df['radius'] / r_tot, df['vel_3_rms'], label=df['time'][0])
    ax2.set_title(i)

# ax.legend()
ax1.legend()
ax2.legend()

timestr = strftime("%d-%b-%Y_%H-%M-%S%p", gmtime())

fig.savefig(f"log_vel_1_rms_{timestr}.png")
fig1.savefig(f"vel_2_rms_{timestr}.png")
fig2.savefig(f"vel_3_rms_{timestr}.png")