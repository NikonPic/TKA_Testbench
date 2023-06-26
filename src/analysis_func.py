
# %%
from tqdm import tqdm
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import signal
import quaternion
from pyquaternion import Quaternion
import scipy.io
from numpy.core.multiarray import ndarray
import numpy as np
import math

# %%


def read_dir(infos: dict):
    """
    Extract all files from a repository
    :param infos: all relevant infos about the data
    :return the raw data
    """
    raw_data_robot = []
    raw_data_tracking = []
    data_count = 0

    # collect the robot data
    print('Reading all ', infos['extension'], 'files')
    for file in tqdm(os.listdir(infos['dir'])):
        if file.endswith(infos['extension']):
            filename = infos['dir'] + "/" + file

            # Select the category:
            for cat in infos['categories']:
                # Check if filename contains the category
                if cat in filename:
                    # now open the file and extract the data
                    local_data = read_testfile_to_data(filename, infos['deg_arr'], infos['start'],
                                                       infos['but_ord'],
                                                       infos['but_freq'], cat)
                    data_count += 1
                    raw_data_robot.append(local_data)

    # collect the tracker data
    print('Reading all ', infos['extension_tr'], 'files')
    for file in tqdm(os.listdir(infos['dir_tr'])):
        if file.endswith(infos['extension_tr']):
            filename = infos['dir_tr'] + "/" + file

            # Select the category:
            for cat in infos['categories']:
                # Check if filename contains the category
                if cat in filename:
                    local_data = tracker_to_pd(filename, infos)
                    data_count += 1
                    raw_data_tracking.append(local_data)

    return raw_data_robot, raw_data_tracking


def tracker_to_pd(filename: str, infos: dict) -> dict:
    """
    generate the tracking dictionary
    """
    # starting number
    start = infos['start']

    # get raw tracking data
    tracking = pd.read_csv(filename)

    # genrate the matrices base and tool from it:
    base = tracking.iloc[start:, 0:7].to_numpy()

    # get the tool matrix
    tool = tracking.iloc[start:, 7:14].to_numpy()

    # extract the time vector
    time = tracking.loc[start:, 'time'].to_numpy()

    # build dictionary
    tr = {"base": base, "tool": tool, "time": time}

    return tr


def max_arr(arr, ran=50):
    """
    Take array arr and determine its mean max_position
    """
    sh = arr.shape
    min_pos = np.argmax(arr)  # Get max position
    # determine the range within array
    mean_range = range(max(min_pos - ran, 0), min(min_pos + ran, sh[0]))
    # collect the data around the max
    mean_max = np.mean(arr[mean_range])

    return mean_max


def fit_degree(data, deg_arr):
    """
    As the testing was done with various angles this script
    evaluates the match towards the relevant angles.
    """
    # The first stop array:
    s_arr = (data['s_app'] == 0)
    # Determine length of first stopping period
    first_s = np.where(s_arr == 0)[0][0]
    # Calculate mean angle during that period
    init_angle = np.mean(data['eul'][range(first_s), 2])
    # get the position in the degree array
    diff_p = np.argmin(np.abs(deg_arr - init_angle))
    # get the actual starting angle:
    deg_start = deg_arr[diff_p]
    # correct the angle vector:
    diff = (init_angle - deg_start)
    data['offset'][2] += diff
    data['eul'][:, 2] -= (deg_start - init_angle)


def to_quat_array(arr):
    """primitive function to stack the pyquaternions"""
    q_arr = []
    for element in arr:
        q_arr.append(Quaternion(element))

    return q_arr


def as_euler_angles(arr):
    """transform pyquaternions to yaw pitch roll"""
    euler_arr = np.empty([len(arr), 3])

    for i, quat in enumerate(arr):
        yaw, pitch, roll = quat.yaw_pitch_roll
        euler_arr[i, :] = np.array([yaw, pitch, roll])

    return euler_arr


def read_testfile_to_data(filename: str, deg_arr, start, but_ord=4, but_freq=0.03, cat='unknown', quat_py=False) -> object:
    """
    This function does the job of reading the input data and returning a
    data struct with all relevant data.
    """
    # Load mat data
    mat = scipy.io.loadmat(filename)
    exp_data = mat['exp_data']

    # The Movement:
    x_act = exp_data['X_act'][0][0]
    pos = x_act[start:, range(3)]  # Positions
    rot = x_act[start:, range(3, 7)]  # Rotations

    # using quaternion
    quat = quaternion.as_quat_array(rot)
    eul = quaternion.as_euler_angles(quat)

    # Remove initial offset
    off_range = 1000
    save_offset = [np.mean(eul[1:off_range, 0]), np.mean(
        eul[1:off_range, 1]), np.mean(eul[1:off_range, 2])]

    eul[:, 0] -= np.mean(eul[1:off_range, 0])
    eul[:, 1] -= np.mean(eul[1:off_range, 1])
    eul[:, 2] -= np.mean(eul[1:off_range, 2])

    # Now filtering is possible
    # define the angles to be within -π / π
    eul += (eul < -math.pi) * 2 * math.pi
    eul *= (180 / math.pi)  # transform to degrees

    eul[:, 2] -= max_arr(eul[:, 2]) - np.max(deg_arr)

    #

    # The Forces and Moments:
    ft_act = exp_data['FT_act'][0][0][start:, :]

    # The applications:
    s_app = exp_data['s_application'][0][0][start:, :]

    # The control
    s_con = exp_data['s_control'][0][0][start:, :]

    # the time vector:
    time = exp_data['Time'][0][0][start:, :]

    """
	Filter the Data - Using Butterworth
	"""
    # Filter Constants
    b, a = signal.butter(but_ord, but_freq)

    # Filter Positions:
    # filtfilt -> no delay occurs!
    pos_f = signal.filtfilt(b, a, np.transpose(pos))
    pos_f = np.transpose(pos_f)

    # Filter Angles:
    eul_f = signal.filtfilt(b, a, np.transpose(eul))
    eul_f = np.transpose(eul_f)

    # Filter Forces and Torques:
    ft_act = signal.filtfilt(b, a, np.transpose(ft_act))
    ft_act = np.transpose(ft_act)

    data = {
        'quat2': rot,
        'quat': quat,
        'pos': pos_f,
        'eul': eul_f,
        'FT_act': ft_act,
        's_app': s_app,
        's_con': s_con,
        'offset': save_offset,
        'cat': cat,
        'time': time,
    }

    fit_degree(data, deg_arr)

    return data


def copy_data(data, ranging, p, deg):
    """
    This function copies all relevant infos from data to data_loc
    :param data: the input data
    :param p: the selected program
    :param ranging: the information needed
    :param, deg: the matching degree
    :return: data_loc
    """
    data_loc = {
        'quat': data['quat'][ranging],
        'pos': data['pos'][ranging],
        'eul': data['eul'][ranging],
        'FT_act': data['FT_act'][ranging],
        's_app': data['s_app'][ranging],
        's_con': data['s_con'][ranging],
        'offset': data['offset'],
        'program': p,
        'deg': deg,
        'cat': data['cat']
    }
    return data_loc


def partitioning(data: dict, deg_arr, program_info: dict):
    """
    Split the data according to their Programs and Angles
    """
    # Take the application data
    s_app = data['s_app']
    eul = data['eul']
    # Detect value changes
    changes = s_app[:-1] != s_app[1:]
    # Take indices of all value changes
    tasks_arr = np.where(changes == 1)[0]
    # Preallocate size:
    task_nr = tasks_arr.shape[0]

    # Extract the values from the dictionary:
    prog_values = program_info.values()

    # Iterate over all different tasks
    pos_o = 0
    # create empyt list
    parts = []

    for going in range(task_nr):
        # Get the local range
        r_loc = range(pos_o + 1, tasks_arr[going])
        # Get the local programm:
        p_loc = s_app[pos_o + 1]
        # Get the local Angle!
        eul_loc = np.mean(eul[r_loc, 2])
        # match the local angle:
        mindiff_pos = np.argmin(np.abs(deg_arr - eul_loc))
        # get the active angle:
        angle_loc = deg_arr[mindiff_pos]

        # only build new array it the current program is within the required data
        if p_loc in prog_values:
            data_loc = copy_data(data, r_loc, p_loc, angle_loc)
            parts.append(data_loc)

        # update the last position:
        pos_o = tasks_arr[going]

    return parts


def remove_offset(off_range, a, b, c):
    a -= np.mean(a[1:off_range])
    b -= np.mean(b[1:off_range])
    c -= np.mean(c[1:off_range])


def get_col(cat):
    """color definitions for plotting"""
    switcher = {
        "nativ": 'g',
        "+5": 'r',
        "-5": 'b',
        "+-0": 'c',
    }
    return switcher.get(cat, "Invalid name")


def zero_quat(part, infos):
    """correct quat and return euler angles"""
    quat = part['quat']
    quat = quat * infos['q_change']
    eul = quaternion.as_euler_angles(quat)
    eul += (eul < -math.pi) * 2 * math.pi
    eul *= (180 / math.pi)  # transform to degrees
    alp, bet, gam = eul[:, 0], eul[:, 1], eul[:, 2]
    return alp, bet, gam


def get_rot_angle(p):

    quat = p['quat']
    q1 = quat[0]
    theta = np.empty([len(quat)])

    for i, q2 in enumerate(quat):
        theta[i] = get_min_angle(q1, q2)

    return theta


def get_min_angle(q1, q2):
    """
    Minimum angle between quaternions acc to :
    https://www.researchgate.net/post/How_do_I_calculate_the_smallest_angle_between_two_quaternions
    """
    q1q2_conj = q1 * q2.conjugate()
    e = np.array([q1q2_conj.x, q1q2_conj.y, q1q2_conj.z])
    e_norm = np.linalg.norm(e)
    theta = 2 * math.asin(e_norm) * (180 / math.pi)
    return theta


def get_quat_rom(p):
    """
    Get the range of motion (ROM) depending on the extreme states.
    """
    quaternions = p['quat']
    state_conditions = p['s_con']
    current_state = state_conditions[0]
    range_of_motion = []
    quat_pair = []
    state_transition_count = 0

    # Iterate over state_conditions and quaternions
    for state, quat in zip(state_conditions, quaternions):
        # Continue if state is unchanged
        if state == current_state:
            continue
        
        # Update current_state
        current_state = state

        # Skip state 31
        if current_state == 31:
            continue

        state_transition_count += 1

        # Skip every second state transition
        if state_transition_count == 2:
            state_transition_count = 0
            continue

        # Save the quaternion
        quat_pair.append(quat)

        # Calculate ROM when two quaternions are available
        if len(quat_pair) >= 2:
            angle = get_min_angle(quat_pair[0], quat_pair[1])
            range_of_motion.append(angle)
            # Empty quat_pair
            quat_pair = []

    return range_of_motion


def get_quat_inex_varvalg(p):
    """
    get the rom depending on the extreme states
    """
    # get starting cond.
    quat = p['quat']
    s_con = p['s_con']
    s_cur = s_con[0]

    var_in = []
    val_ex = []

    count = 0

    q12 = []
    q12.append(quat[0])

    varin = True

    # iterate
    for s, q in zip(s_con, quat):
        # continue if unchanged
        if s == s_cur:
            continue
        # set to new state
        s_cur = s

        if s_cur == 31:
            continue

        count += 1

        if count == 2:
            count = 0
            continue

        # save the quaternion
        q12.append(q)

        if len(q12) >= 2:
            theta = get_min_angle(q12[0], q12[1])

            q12 = []
            q12.append(quat[0])

            if varin:
                var_in.append(theta)
                varin = False

            else:
                val_ex.append(theta)
                varin = True

    return var_in, val_ex


def generate_ROM(parts: dict, infos: dict, quat_calc=True, use_all=True, key='eul'):
    """
    first attain the maximum ROM over alp, bet, gam and save it in a new dict -> array
    so: dat_rom['nativ']['10'] = [first, second, third]
    """

    dat_rom = {}

    for p in parts:

        # create empty dict...
        if p['cat'] not in dat_rom.keys():
            dat_rom[p['cat']] = {}

        # create empty array
        if str(p['deg']) not in dat_rom[p['cat']].keys():
            dat_rom[p['cat']][str(p['deg'])] = {}
            dat_rom[p['cat']][str(p['deg'])]['ang'] = []
            dat_rom[p['cat']][str(p['deg'])]['max'] = []

            dat_rom[p['cat']][str(p['deg'])]['var_in'] = []
            dat_rom[p['cat']][str(p['deg'])]['valg_ex'] = []

        # now extract the ROM data of p:
        alp, bet, gam = [p[key][:, 0], p[key][:, 1], p[key][:, 2]]
        rom_alp = np.max(alp) - np.min(alp)
        rom_bet = np.max(bet) - np.min(bet)
        rom_gam = np.max(gam) - np.min(gam)
        rom_arr = np.array([rom_alp, rom_bet, rom_gam])
        rom_max = np.sqrt(np.sum(np.square(rom_arr))) * 100

        if len(dat_rom[p['cat']][str(p['deg'])]['max']) == 0 or use_all:
            # append to the data struct
            dat_rom[p['cat']][str(p['deg'])]['ang'].append(rom_arr)

            if quat_calc:
                rom_max = get_quat_rom(p)
                var_in, val_ex = get_quat_inex_varvalg(p)
                dat_rom[p['cat']][str(p['deg'])]['max'].extend(rom_max)
                dat_rom[p['cat']][str(p['deg'])]['var_in'].extend(var_in)
                dat_rom[p['cat']][str(p['deg'])]['valg_ex'].extend(val_ex)

            else:
                # append to the data struct
                dat_rom[p['cat']][str(p['deg'])]['max'].append(rom_max)

    return dat_rom


def gen_plot_dict(dat_rom, infos):
    """
    ok now we have all data
    go trough all  keys and the mean and var
    """
    # dict to fill
    plot_dict = {}

    # translate between degrees and array index
    deg_order = {'10.0': 0, '20.0': 1, '30.0': 2, '60.0': 3, '90.0': 4}

    for cat in dat_rom.keys():
        # sub -dict
        plot_dict[cat] = {}
        plot_dict[cat]['m_ang'] = np.zeros([5, 3])
        plot_dict[cat]['std_ang'] = np.zeros([5, 3])
        plot_dict[cat]['m_max'] = np.zeros([5])
        plot_dict[cat]['std_max'] = np.zeros([5])

        plot_dict[cat]['var_in'] = np.zeros([5])
        plot_dict[cat]['std_var_in'] = np.zeros([5])

        plot_dict[cat]['valg_ex'] = np.zeros([5])
        plot_dict[cat]['std_valg_ex'] = np.zeros([5])

        for deg in dat_rom[cat].keys():
            # get mean and std for each:
            ang = np.array(dat_rom[cat][deg]['ang'])
            max_ang = np.array(dat_rom[cat][deg]['max'])

            var_in = np.array(dat_rom[cat][deg]['var_in'])
            valg_ex = np.array(dat_rom[cat][deg]['valg_ex'])

            if len(ang[0]) > 1:
                ang = ang[0]
            mean_ang = np.mean(ang, axis=0)
            std_ang = np.std(ang, axis=0)

            mean_max_ang = np.mean(max_ang, axis=0)
            std_max_ang = np.std(max_ang, axis=0)

            mean_var_in = np.mean(var_in, axis=0)
            std_var_in = np.std(var_in, axis=0)

            mean_valg_ex = np.mean(valg_ex, axis=0)
            std_valg_ex = np.std(valg_ex, axis=0)

            # append to dict
            plot_dict[cat]['m_ang'][deg_order[deg], :] = mean_ang
            plot_dict[cat]['std_ang'][deg_order[deg], :] = std_ang

            plot_dict[cat]['m_max'][deg_order[deg]] = mean_max_ang
            plot_dict[cat]['std_max'][deg_order[deg]] = std_max_ang

            plot_dict[cat]['var_in'][deg_order[deg]] = mean_var_in
            plot_dict[cat]['std_var_in'][deg_order[deg]] = std_var_in

            plot_dict[cat]['valg_ex'][deg_order[deg]] = mean_valg_ex
            plot_dict[cat]['std_valg_ex'][deg_order[deg]] = std_valg_ex

    return plot_dict


def plotfill(x_arr, alp, std, col, cat, std_fac):
    """short custom func for -gen_rom_plots-"""
    idx = np.nonzero(alp)[0]
    plt.plot(x_arr[idx], alp[idx], color=col, label=cat)
    plt.fill_between(x_arr[idx], alp[idx] + std_fac*std[idx],
                     alp[idx] - std_fac*std[idx], color=col, alpha=0.25)
    #plt.legend()

def heler_plot_angles():
    plt.figure('Alpha ROM')
    plt.clf()
    plt.xlabel("Felxion Angle [°]")
    plt.ylabel("ROM alpha [°]")
    plt.grid(0.25)

    plt.figure('Beta ROM')
    plt.clf()
    plt.xlabel("Felxion Angle [°]")
    plt.ylabel("ROM beta [°]")
    plt.grid(0.25)

    plt.figure('Gamma ROM')
    plt.clf()
    plt.xlabel("Felxion Angle [°]")
    plt.ylabel("ROM gamma [°]")
    plt.grid(0.25)

def gen_rom_plots(plot_dict, infos, std_fac=1, show_angles=False):
    x_arr = np.array([10, 20, 30, 60, 90])

    if show_angles:
        heler_plot_angles()

    if 'Varus-Valgus' in infos['rel_prog'].keys():
        xlab1 = "Varus [°]"
        xlab2 = "Valgus [°]"
    else:
        xlab1 = "Internal [°]"
        xlab2 = "External [°]"
    
    plt.figure('Maximum ROM')
    plt.clf()
    plt.xlabel("Felxion Angle [°]")
    plt.ylabel("ROM [°]")
    plt.grid(0.25)

    plt.figure('Maximum Varus / Iternal')
    plt.clf()
    plt.xlabel("Felxion Angle [°]")
    plt.ylabel(xlab1)
    plt.grid(0.25)

    plt.figure('Maximum Valgus / External')
    plt.clf()
    plt.xlabel("Felxion Angle [°]")
    plt.ylabel(xlab2)
    plt.grid(0.25)

    plt.figure('Score')
    plt.clf()
    plt.xlabel("Flexion Angle [°]")
    plt.ylabel("Score []")

    for cat in plot_dict.keys():
        col = get_col(cat)

        plt.figure('Maximum ROM')
        alp = plot_dict[cat]['m_max']
        std = plot_dict[cat]['std_max']
        plotfill(x_arr, alp, std, col, cat, std_fac)
        plt.legend()

        plt.figure('Maximum Varus / Iternal')
        var_in = plot_dict[cat]['var_in']
        std = plot_dict[cat]['std_var_in']
        plotfill(x_arr, var_in, std, col, cat, std_fac)
        plt.legend()

        plt.figure('Maximum Valgus / External')
        valg_ex = plot_dict[cat]['valg_ex']
        std = plot_dict[cat]['std_valg_ex']
        plotfill(x_arr, valg_ex, std, col, cat, std_fac)
        plt.legend()

        if not show_angles:
            continue

        plt.figure('Alpha ROM')
        alp = plot_dict[cat]['m_ang'][:, 0]
        std = plot_dict[cat]['std_ang'][:, 0]
        plotfill(x_arr, alp, std, col, cat, std_fac)

        plt.figure('Beta ROM')
        alp = plot_dict[cat]['m_ang'][:, 1]
        std = plot_dict[cat]['std_ang'][:, 1]
        plotfill(x_arr, alp, std, col, cat, std_fac)

        plt.figure('Gamma ROM')
        alp = plot_dict[cat]['m_ang'][:, 2]
        std = plot_dict[cat]['std_ang'][:, 2]
        plotfill(x_arr, alp, std, col, cat, std_fac)

        plt.figure('Maximum ROM')
        alp = plot_dict[cat]['m_max']
        std = plot_dict[cat]['std_max']
        plotfill(x_arr, alp, std, col, cat, std_fac)
        plt.legend()

    # ensure last ylimit is aligned with zero
    plt.figure('Maximum ROM')
    plt.ylim(0, plt.ylim()[1])

    plt.figure('Maximum Varus / Iternal')
    plt.title("Applied Momentum: 5 Nm")
    max1 = plt.ylim()[1]
    plt.ylim(0, max1)

    plt.figure('Maximum Valgus / External')
    plt.title("Applied Momentum: 5 Nm")
    max1 = max(max1, plt.ylim()[1])
    plt.ylim(0, max1)

    plt.figure('Maximum Varus / Iternal')
    plt.ylim(0, max1)


def rom_plots(parts, infos, use_all=True, std_fac= 1.0):
    """summary of rom generation"""
    dat_rom = generate_ROM(parts, infos, use_all=use_all)
    plot_dict = gen_plot_dict(dat_rom, infos)
    gen_rom_plots(plot_dict, infos, std_fac=std_fac)
    return plot_dict


def show_rot(parts: dict, infos: dict, q_change, offset=True):
    """display rotations only"""

    new_fig3 = plt.figure("show rot", figsize=(16, 8))
    new_fig3.clf()

    for p in parts:
        off_range = 100

        # Only visualize if the data is within the relevant degrees:
        if p['deg'] in infos['rel_deg'] and p['cat'] in infos['rel_cat']:

            quat = p['quat']
            quat = quat * q_change

            eul = quaternion.as_euler_angles(quat)

            # Now filtering is possible
            # define the angles to be within -π / π
            eul += (eul < -math.pi) * 2 * math.pi
            eul *= (180 / math.pi)  # transform to degrees
            # eul[:, 2] -= max_arr(eul[:, 2]) - np.max(deg_arr)

            alp, bet, gam = eul[:, 0], eul[:, 1], eul[:, 2]

            if offset:
                remove_offset(off_range, alp, bet, gam)

            plt.subplot(1, 3, 1)
            plt.plot(alp, bet)
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.xlabel('alp')
            plt.ylabel('bet')
            plt.grid(0.25)

            plt.subplot(1, 3, 2)
            plt.plot(alp, gam)
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.xlabel('alp')
            plt.ylabel('gam')
            plt.grid(0.25)

            plt.subplot(1, 3, 3)
            plt.plot(bet, gam)
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.xlabel('bet')
            plt.ylabel('gam')
            plt.grid(0.25)

    plt.show()


def vis_partitioning(parts: dict, infos: dict, offset=True, disp_forces=False, zero_quats=True, show_all=False):
    """
    Check if the partitioning actually worked properly
    :param parts: the dictionary containing the relevant data
    :param infos: contains constraints for the relevant parts
    :return:
    """
    title_name = list(infos['rel_prog'].keys())[:]

    new_fig2 = plt.figure("show positions")
    new_fig2.clf()
    ax2 = new_fig2.add_subplot(projection='3d')
    plt.title(title_name)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    new_fig3 = plt.figure("show rotations")
    new_fig3.clf()
    ax3 = new_fig3.add_subplot(projection='3d')
    plt.title(title_name)
    ax3.set_xlabel("alpha [°]")
    ax3.set_ylabel("beta [°]")
    ax3.set_zlabel("gamma [°]")

    if disp_forces:
        new_fig4 = plt.figure("show forces")
        new_fig4.clf()
        ax4 = new_fig4.add_subplot(projection='3d')
        plt.title(title_name)
        ax4.set_xlabel("Fx [N]")
        ax4.set_ylabel("Fy [N]")
        ax4.set_zlabel("Fz [N]")

    new_fig5 = plt.figure("show momentum")
    new_fig5.clf()
    ax5 = new_fig5.add_subplot(projection='3d')
    plt.title(title_name)
    ax5.set_xlabel("Mx [Nm]")
    ax5.set_ylabel("My [Nm]")
    ax5.set_zlabel("Mz [Nm]")

    new_fig12 = plt.figure("Momentum-angles")
    new_fig12.clf()
    ax12 = new_fig12.add_subplot(projection='3d')
    plt.title(title_name)
    ax12.set_xlabel("alpha [°]")
    ax12.set_ylabel("beta [°]")
    ax12.set_zlabel("M [Nm]")

    # Moments
    # =============================================
    new_fig6 = plt.figure("Hyserese moment-alpha")
    new_fig6.clf()
    plt.title(title_name)
    plt.grid(0.25)
    plt.xlabel('alpha [°]')
    plt.ylabel('Momentum [Nm]')

    new_fig7 = plt.figure("Hyserese moment-beta")
    new_fig7.clf()
    plt.title(title_name)
    plt.grid(0.25)
    plt.xlabel('beta [°]')
    plt.ylabel('Momentum [Nm]')

    new_fig8 = plt.figure("Hyserese moment-gamma")
    new_fig8.clf()
    plt.title(title_name)
    plt.grid(0.25)
    plt.xlabel('gamma[°]')
    plt.ylabel('Momentum [Nm]')
    # =============================================

    if disp_forces:
        # Forces
        # =============================================
        new_fig9 = plt.figure("Hyserese force-alpha")
        new_fig9.clf()
        plt.title(title_name)
        plt.grid(0.25)
        plt.xlabel('alpha [°]')
        plt.ylabel('Force [N]')

        new_fig10 = plt.figure("Hyserese force-beta")
        new_fig10.clf()
        plt.title(title_name)
        plt.grid(0.25)
        plt.xlabel('beta [°]')
        plt.ylabel('Force [N]')

        new_fig11 = plt.figure("Hyserese force-gamma")
        new_fig11.clf()
        plt.title(title_name)
        plt.grid(0.25)
        plt.xlabel('gamma[°]')
        plt.ylabel('Force [N]')
        # =============================================

    run_x = 0

    infos_parts = {}

    for p in parts:
        # update the local range
        x_range = range(run_x, run_x + len(p['s_app']))
        run_x += len(p['s_app'])

        off_range = 100

        # Only visualize if the data is within the relevant degrees:
        if p['deg'] in infos['rel_deg'] and p['cat'] in infos['rel_cat']:

            heading = str(int(p["deg"])) + "° " + p['cat']
            col = get_col(p['cat'])

            # plot the relevant position
            plt.figure("show positions")
            # assign positions
            x, y, z = p["pos"][:, 0].copy(
            ), p["pos"][:, 1].copy(), p["pos"][:, 2].copy()

            if offset:
                remove_offset(off_range, x, y, z)

            if heading not in infos_parts.keys() or show_all:
                plt.plot(x, y, z, label=heading, color=col)
                plt.legend()

            # plot the relevant position
            plt.figure("show rotations")

            # assign positions
            if zero_quats:
                alp, bet, gam = zero_quat(p, infos)
            else:
                alp, bet, gam = p["eul"][:, 0].copy(
                ), p["eul"][:, 1].copy(), p["eul"][:, 2].copy()

            if offset:
                remove_offset(off_range, alp, bet, gam)

            if heading not in infos_parts.keys() or show_all:
                plt.plot(alp, bet, gam, color=col, label=heading)
                plt.legend()

            # plot the relevant forces
            if disp_forces:
                plt.figure("show forces")
            # assign forces
            fx, fy, fz = p["FT_act"][:,
                                     0].copy(), p["FT_act"][:, 1].copy(), p["FT_act"][:, 2].copy()
            if offset:
                remove_offset(off_range, fx, fy, fz)

            if heading not in infos_parts.keys() or show_all and disp_forces:
                plt.plot(fx, fy, fz, label=heading, color=col)
                plt.legend()

            # plot the relevant momentum
            plt.figure("show momentum")
            # assign forces
            mx, my, mz = p["FT_act"][:, 3].copy(
            ), p["FT_act"][:, 4].copy(), p["FT_act"][:, 5].copy()
            if offset:
                remove_offset(off_range, mx, my, mz)

            if heading not in infos_parts.keys() or show_all:
                plt.plot(mx, my, mz, label=heading, color=col)
                plt.legend()

            # -----------------------------------------------------------
            # -----------------------------------------------------------
            # plot the hysterese curves

            # Moments over angles
            M_all = np.sqrt(np.square(np.transpose(
                np.array([mx, my, mz]))).sum(axis=1))

            # plot the relevant momentum
            plt.figure("Momentum-angles")
            plt.plot(alp, bet, M_all, label=heading, color=col)
            plt.legend()

            if heading not in infos_parts.keys():
                plt.figure("Hyserese moment-alpha")
                plt.plot(alp, M_all, label=heading, color=col)
                plt.legend()

                plt.figure("Hyserese moment-beta")
                plt.plot(bet, M_all, label=heading, color=col)
                plt.legend()

                plt.figure("Hyserese moment-gamma")
                plt.plot(gam, M_all, label=heading, color=col)
                plt.legend()

            if disp_forces:
                # Forces over angles
                F_all = np.sqrt(np.square(np.transpose(
                    np.array([fx, fy, fz]))).sum(axis=1))
                plt.figure("Hyserese force-alpha")
                plt.plot(alp, F_all, label=heading, color=col)
                plt.legend()

                plt.figure("Hyserese force-beta")
                plt.plot(bet, F_all, label=heading, color=col)
                plt.legend()

                plt.figure("Hyserese force-gamma")
                plt.plot(gam, F_all, label=heading, color=col)
                plt.legend()

            if heading not in infos_parts.keys():
                infos_parts[heading] = {}
                infos_parts[heading]["alpha"] = np.array([])
                infos_parts[heading]["beta"] = np.array([])
                infos_parts[heading]["M_all"] = np.array([])

            infos_parts[heading]["alpha"] = np.append(
                infos_parts[heading]["alpha"], alp)
            infos_parts[heading]["beta"] = np.append(
                infos_parts[heading]["beta"], bet)
            infos_parts[heading]["M_all"] = np.append(
                infos_parts[heading]["M_all"], M_all)

            fig = plt.figure("Moment Surface " + heading)
            fig.clf()
            ax = fig.add_subplot(projection='3d')

            ax.plot_trisurf(
                infos_parts[heading]["beta"],
                infos_parts[heading]["alpha"],
                infos_parts[heading]["M_all"],
                linewidth=0.2, antialiased=True,
            )

            ax.set_ylabel("alpha")
            ax.set_xlabel("beta")
            ax.set_zlabel("Moment [Nm]")
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.set_zlim([0, 5])


def show_all(data):
    """
    Extracting the movement and visualising it
    """
    # Take from data:
    pos = data['pos']
    eul = data['eul']

    # Plot the movement
    fig = plt.figure("Movement")
    plt.clf()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    # Define coordinates of movement
    x, y, z = (pos[:, 0], pos[:, 1], pos[:, 2])
    ax.plot(x, y, z, label='Movement of the TCP')
    ax.legend()

    # Now filter again:
    fig2 = plt.figure("Rotation")
    plt.clf()
    ax2 = fig2.gca(projection='3d')
    ax2.set_aspect('equal')
    # Define coordinates of rotation
    alp, bet, gam = eul[:, 0], eul[:, 1], eul[:, 2]
    ax2.plot(alp, bet, gam, label='Rotation of the TCP')


def partition_all(all_data: list, infos: dict):
    """
    Partition all data according to their matching program, category, degree
    :param all_data: Contains all relevant data from directory
    :param infos: contains all data from the files
    :return: all partitioned subparts
    """
    # Initialise
    all_parts = []
    # Go over all files:
    for dat in all_data:
        local_parts = partitioning(dat, infos['deg_arr'], infos['rel_prog'])
        # Extend only if data points actually occured!
        if len(local_parts) > 0:
            all_parts.extend(local_parts)
    # Return
    return all_parts

# %%


if __name__ == '__main__':
    pass

# %%
