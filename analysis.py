# %%
import csv
"""
Analysis Script of the testing run from 14.12.2018 of the Human Specimen
Imports and General Parameters

Quick documentationn on the available Programs:
control_states = [0,1,2,3,5,14,19,23,24,25,26,27];
(0):  Stop
(1):  TrackingCalibration
(2):  FTSCalibration
(3):  ConnectSpecimen
(5):  Return Home
(14): Handmove
(19): Flexion
(23): Pivot Shift
(24): Varus-Valgus Loading
(25): Internal-External Loading
(26): Lachmann Loading
(27): Lachmann Loading v2
"""
import matplotlib.pyplot as plt  # Plots

import numpy as np
import scipy
from time import sleep
from tqdm import tqdm
import pandas as pd

import src.analysis_func as af
import src.fuse_tracking as ft

import quaternion
 #%matplotlib qt

q_offset = np.quaternion(1, 0, 0, 0)
q_offset1 = np.quaternion(0.3, -0.3, 0.2, 0.2)

# Important Parameters
infos = {
    # Important Infos which data to take
    'dir': './data/robot',  # path of important robot files
    'dir_tr': './data/tracking',
    'extension': 'mat',  # data extension
    'extension_tr': 'csv',  # tracking data extension
    'categories': ["nativ", "+-0", "+5", "-5"],  # categories

    # Information about the preparation of the data
    'start': 100,  # Starting parameter of visualization
    'fs': 100,  # Control Frequency in Hz
    'but_ord': 4,  # Order of Butterworth filter
    'but_freq': 0.5,  # Cutoff frequency
    'deg_arr': [10.0, 20.0, 30.0, 60.0, 90.0],  # Tested degrees

    # Information about which data to select!
    'rel_deg': [60.0],  # the relevant data of the degrees!
    'rel_cat': ["nativ", "+-0" , "_th_", "_pld_"],
    'rel_prog': {
        # 'Stop':0,
        # 'Flexion':19,
        # 'Pivot Shift':23,
         'Varus-Valgus': 24,
        # 'Internal-External': 25,
        # 'Lachmann': 26,
        # 'Lachmann v2':27,
    },
    'q_change': q_offset, # offset quaternion
}

def get_norm(arr):
    return [np.linalg.norm(ele) for ele in arr]

def prepare():
    """
    Load the data from diretory
    Fuse the robot and tracking data
    """
    rob_data, tr_data = af.read_dir(infos)
    rob_data, tr_data = ft.fuse_rob_tr(rob_data, tr_data, infos)
    return rob_data, tr_data


# %%
if __name__ == "__main__":
    # load files
    rob_data, tr_data = af.read_dir(infos)
    # partition files
    all_parts = af.partition_all(rob_data, infos)
    # visualize files
    af.vis_partitioning(all_parts, infos)

# %%
plot_dict = af.rom_plots(all_parts, infos, use_all=False)
plot_dict
# %%
def gen_rom_plots(plot_dict, infos, std_fac=1, show_angles=False):
    x_arr = np.array([10, 20, 30, 60, 90])


    if 'Varus-Valgus' in infos['rel_prog'].keys():
        title1 = "5 Nm load varus"
        title2 = "5 Nm load valgus"
    else:
        title1 = "5 Nm load internal"
        title2 = "5 Nm load external"
    
    plt.figure('Maximum ROM')
    plt.clf()
    plt.xlabel("felxion angle in °")
    plt.ylabel("ROM in °")
    plt.grid(0.25)

    plt.figure('Maximum Varus / Iternal')
    plt.clf()
    plt.subplot(121)
    plt.title(title1)
    plt.xlabel("felxion angle in °")
    plt.ylabel("deviation in °")
    plt.grid(0.25)

    plt.subplot(122)
    plt.title(title2)
    plt.xlabel("felxion angle in °")

    plt.grid(0.25)

    plt.figure('Score')
    plt.clf()
    plt.xlabel("felxion angle in °")
    plt.ylabel("Score []")

    for cat in plot_dict.keys():
        col = af.get_col(cat)

        plt.figure('Maximum ROM')
        alp = plot_dict[cat]['m_max']
        std = plot_dict[cat]['std_max']
        af.plotfill(x_arr, alp, std, col, cat, std_fac)
        plt.legend()

        plt.figure('Maximum Varus / Iternal')
        plt.subplot(121)
        var_in = plot_dict[cat]['var_in']
        std = plot_dict[cat]['std_var_in']
        af.plotfill(x_arr, var_in, std, col, cat, std_fac)

        plt.subplot(122)
        valg_ex = plot_dict[cat]['valg_ex']
        std = plot_dict[cat]['std_valg_ex']
        af.plotfill(x_arr, valg_ex, std, col, cat, std_fac)
        plt.legend()

        plt.figure('Alpha ROM')
        alp = plot_dict[cat]['m_ang'][:, 0]
        std = plot_dict[cat]['std_ang'][:, 0]
        af.plotfill(x_arr, alp, std, col, cat, std_fac)

        plt.figure('Beta ROM')
        alp = plot_dict[cat]['m_ang'][:, 1]
        std = plot_dict[cat]['std_ang'][:, 1]
        af.plotfill(x_arr, alp, std, col, cat, std_fac)

        plt.figure('Gamma ROM')
        alp = plot_dict[cat]['m_ang'][:, 2]
        std = plot_dict[cat]['std_ang'][:, 2]
        af.plotfill(x_arr, alp, std, col, cat, std_fac)

        plt.figure('Maximum ROM')
        alp = plot_dict[cat]['m_max']
        std = plot_dict[cat]['std_max']
        af.plotfill(x_arr, alp, std, col, cat, std_fac)
        plt.legend()

    # ensure last ylimit is aligned with zero
    plt.figure('Maximum ROM')
    plt.ylim(0, plt.ylim()[1])

    plt.figure('Maximum Varus / Iternal')
    plt.subplot(121)
    max1 = plt.ylim()[1]
    plt.ylim(0, max1)
    
    plt.subplot(122)
    max1 = max(max1, plt.ylim()[1])
    plt.ylim(0, max1)

    plt.tight_layout()

    plt.figure('Maximum Varus / Iternal')
    plt.ylim(0, max1)

# %%
gen_rom_plots(plot_dict, infos, show_angles=True)
# %%
#all_parts[0].keys()
#plt.plot(get_norm(all_parts[0]['eul']))
# %%





#norm_arr = get_norm(all_parts[22]['pos'])
#plt.plot(norm_arr)
# %%
#all_parts[0].keys()
#plt.plot(get_norm(all_parts[2]['eul']))
plt.show()
# %%
