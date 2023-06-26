import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from scipy import signal
import quaternion


def fuse_rob_tr(rob_data: list, tr_data: list, infos: dict):
    """
    Try to fuse the two data structs to attain one final result!
    :param rob_data: Data containing all relevant robot measurements
    :param tr_data: tracking data
    :param infos: parameters
    :return:
    """

    tr_data = resample_tr(tr_data, infos)  # get on same sampling frequency
    # get the synchronization difference
    diff_all = sync_all(rob_data, tr_data)
    rob_data, tr_data = apply_diff(
        rob_data, tr_data, diff_all)  # move on same sync

    return rob_data, tr_data


def Merge(dict1, dict2):
    """
    merge two dictionaries using ** trick
    """
    res = {**dict1, **dict2}
    return res


def sync_all(rob_data: list, tr_data: list, plot_it=False):
    """
    synchronize all data at once
    """

    # CONSTANTS
    # -----------------------------
    ranging = 0.2  # matching range
    sign = 'pos'  # signal of relevance
    pos_rel = 1  # part of the signal of relevance

    but_ord = 4  # order
    but_freq = 0.01  # high-pass
    # ----------------------------

    # init all differences
    diff_all = np.zeros(len(rob_data))

    for k in tqdm(range(0, len(rob_data))):

        # local data to match
        loc_rob = rob_data[k]
        loc_tr = tr_data[k]

        # get the signals correctly aligned
        tr_pos, rob_pos = arrange(loc_rob, loc_tr, pos_rel=1, sign="pos")

        # highpass filter the data
        rob_pos_f = high_pass(
            rob_pos, but_ord=but_ord, but_freq=but_freq)
        tr_pos_f = high_pass(
            tr_pos, but_ord=but_ord, but_freq=but_freq)

        # extract interesting part
        buffer, position = get_remarkable_part(
            tr_pos_f, ranging=ranging)

        # match robot data
        matching = match_robot(buffer, position, rob_pos_f)

        # apply differences on the signal data
        d = np.argmin(matching) - position[0]

        # save the difference
        diff_all[k] = d

        if plot_it == True:
                # Visualize the process for validation
            plt.figure(k)
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.title("Unsynchronized")
            plt.plot(rob_pos - rob_pos[0], label='robot')
            plt.plot(tr_pos - tr_pos[0], label='tracker')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.title("Synchronized")

            # shift the sync signal
            if d > 0:
                plt.plot(rob_pos[int(d):] - rob_pos[0], label='robot')
                plt.plot(tr_pos - tr_pos[0], label='tracker')
                plt.legend()

            else:
                plt.plot(rob_pos - rob_pos[0], label='robot')
                plt.plot(tr_pos[int(abs(d)):] - tr_pos[0], label='tracker')
                plt.legend()


    return diff_all


def apply_diff(rob_data: list, tr_data: list, diff_all):
    """
    apply the differences between the signals from diff_all
    """
    # go over all collected data
    for k in range(len(diff_all)):
        # get the current distance
        d = int(diff_all[k])
        # local datas
        loc_robot = rob_data[k]
        loc_tr = tr_data[k]

        # robot data ahead
        if d > 0:
            diff_on_dict(loc_robot, d)

        # tracking data ahead
        else:
            diff_on_dict(loc_tr, -d)

        # fit on same length
        cut = max(loc_robot["pos"].shape) - max(loc_tr["pos"].shape)

        # robot data ahead
        if cut > 0:
            cut_on_dict(loc_robot, cut)

        # tracking data ahead
        else:
            cut_on_dict(loc_tr, -cut)

    return rob_data, tr_data


def cut_on_dict(dictio: dict, cut: int):
    """
    apply the difference on all signals of the dictionary!
    """
    # do not change these signals:
    not_allow = ['cat', 'offset']
    for key in dictio.keys():
        if key not in not_allow:
            # get the local data
            loc_data = dictio[key]

            # apply difference
            if len(loc_data.shape) > 1:
                # ensure correct shapes
                if loc_data.shape[0] > loc_data.shape[1]:
                    loc_data = loc_data[:-cut, :]
                else:
                    loc_data = loc_data[:, :-cut + 1]
            else:
                loc_data = loc_data[:-cut + 1]

            # reapply:
            dictio[key] = loc_data


def diff_on_dict(dictio: dict, diff: int):
    """
    apply the difference on all signals of the dictionary!
    """
    # do not change these signals:
    not_allow = ['cat', 'offset']
    for key in dictio.keys():
        if key not in not_allow:
            # get the local data
            loc_data = dictio[key]

            # apply difference
            if len(loc_data.shape) > 1:
                # ensure correct shapes
                if loc_data.shape[0] > loc_data.shape[1]:
                    loc_data = loc_data[diff:, :]
                else:
                    loc_data = loc_data[:, diff:]
            else:
                loc_data = loc_data[diff:]

            # reapply:
            dictio[key] = loc_data


def arrange(loc_rob, loc_tr, pos_rel=0, sign="pos"):
    """
    correctly arrange the dimensions of the selected arrays
    """
    # if rob data has more rows
    if loc_rob[sign].shape[0] > loc_rob[sign].shape[1]:
        rob_pos = loc_rob[sign][:, pos_rel]
    # else more clomuns
    else:
        rob_pos = loc_rob[sign][pos_rel, :]
    # if tr data has more rows
    if loc_tr[sign].shape[0] > loc_tr[sign].shape[1]:
        tr_pos = loc_tr[sign][:, pos_rel]
    # else more columns
    else:
        tr_pos = loc_tr[sign][pos_rel, :]
    return tr_pos, rob_pos


def match_robot(buffer, position, rob_pos):
    """
    Synchronize the robot data with the tracking data
    """
    # get lengths
    buffer_l = len(buffer)
    rob_l = len(rob_pos)

    # init stuff
    loc_buf = np.zeros(buffer_l)
    dist = np.zeros(rob_l - buffer_l)

    # go over the whole trajectory
    for k in range(buffer_l, rob_l):

        # take the local buffer:
        loc_buf[:] = rob_pos[k - buffer_l:k]

        # calculate the distance
        dist[k - buffer_l] = np.linalg.norm(loc_buf - buffer)

    return dist


def get_remarkable_part(test_data, ranging=0.5):
    """
    return a small range with large deviations!
    range is defined in seconds
    pos_rel defines which position sample to use
    """

    # calculate the whole moving range
    true_range = int(len(test_data) * ranging)

    # defines for buffer
    buffer = np.zeros(true_range)
    buffer[:] = test_data[0:true_range]
    min_buffer = np.min(buffer)
    max_buffer = np.max(buffer)
    delta = max_buffer - min_buffer
    highdelta = delta
    change = 0
    position = [0, true_range]

    # now go trough the test of array and find the mins, max etc.
    for k in range(true_range, len(test_data)):

        # analyse old point
        old_point = test_data[k - true_range]

        # get new min
        if old_point == min_buffer:
            min_buffer = np.min(test_data[k - true_range + 1:k + 1])
            change = 1

        # get new max
        if old_point == max_buffer:
            max_buffer = np.max(test_data[k - true_range + 1:k + 1])
            change = 1

        # analyse new point
        new_point = test_data[k]

        # update max?
        if new_point > max_buffer:
            max_buffer = new_point
            change = 1

        # update min?
        if new_point < min_buffer:
            min_buffer = new_point
            change = 1

        # if min or max have changed:
        if change == 1:
            delta = max_buffer - min_buffer
            # update best array
            if delta > highdelta:
                highdelta = delta
                buffer[:] = test_data[k - true_range + 1:k + 1]
                position = [k - true_range + 1, k + 1]

    # return the final buffer!
    return buffer, position


def nan_helper(y):
    """
    Helper to handle indices and logical indices of NaNs.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interp_nan(y):
    """
    linear interpolate over all occuring nans!
    required to apply filters
    """
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def resample_tr(tr_data: list, infos: dict):
    """
    call interpolate_fr from within
    """
    # take frequenzy
    fs = infos['fs']
    # init new data
    new_tr_data = []

    # iterate trough the whole list
    for loc_tr in tr_data:
        old_time = loc_tr['time']
        old_pos = np.transpose(loc_tr['tool'][:, 0:3])
        old_quat = np.transpose(loc_tr['tool'][:, 3:])

        # iterpolate
        new_time, pos_interp, quat_interp = interpol_fr(
            old_time, old_pos, old_quat, fs, interpolate='lin')

        # assign to dictionary
        loc_tr = {
            "time": new_time,
            "pos": pos_interp,
            "quat": quat_interp,
        }

        # append to list
        new_tr_data.append(loc_tr)

    return new_tr_data


def high_pass(sign, but_ord=4, but_freq=0.01):
    """
    make a highpass filter of the signal
    """
    b, a = signal.butter(but_ord, but_freq, btype='high')
    sign = np.nan_to_num(sign, copy=True)
    filt_sign = signal.filtfilt(b, a, sign)
    return filt_sign


def interpol_fr(old_time, old_pos, old_quat, fs, interpolate='lin'):
    """
    interpolate the tracking data to match the sampling frequency
    """
    # build new time vector
    new_time = np.linspace(old_time[0], old_time[-1],
                           int(old_time[-1] * fs), endpoint=False)
    new_length = len(new_time)

    pos_interp = np.zeros([3, new_length], dtype=float, order='C')
    quat_interp = np.zeros([4, new_length], dtype=float, order='C')

    iter_quat = 0

    # Go over all Positions and interpolate normally
    for i in range(0, old_pos.shape[0]):
        # Position can be interpolated normally!
        pos_interp[i, :] = interp_nan(
            np.interp(new_time, old_time, old_pos[i, :]))

    # Interpolate quaternions:
    if interpolate == 'lin':
        for i in range(0, old_quat.shape[0]):
            # Position can be interpolated normally!
            quat_interp[i, :] = interp_nan(
                np.interp(new_time, old_time, old_quat[i, :]))
    else:
        # True SLERP interpolation very costly!
        print('Quaternion interpolation, Number', i)
        for t_loc in tqdm(new_time):
            quat_interp[:, iter_quat] = slerp_it(old_time, old_quat, t_loc)
            iter_quat += 1

    return new_time, pos_interp, quat_interp


def slerp_it(old_time, old_quat, t_loc: float) -> object:
    """
    Idea: get the two closest quaternions to t_animation,
    determine the local t within both!
    this will call slerp from within
    """
    # get the matching time
    t_best = (np.abs(old_time - t_loc)).argmin()

    # select the 2nd best match!
    if t_best == 0:  # Case at start
        t_2nd = 1

    elif t_best == old_time.shape[0] - 1:  # Case at end
        t_2nd = old_time.shape[0] - 2

    else:  # normal case
        neibas = np.array(
            [old_time[t_best - 1], old_time[t_best + 1]])
        best_n = int((np.abs(neibas - t_loc)).argmin() * 2 - 1)
        t_2nd = t_best + best_n

    # get the two quaternions for the matching times:
    q1 = old_quat[:, t_best]
    q2 = old_quat[:, t_2nd]

    # make both quaternions unit quaternions:
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    # get the difference between both matches (dt1 <= dt2)
    dt1 = np.abs(old_time[t_best] - t_loc)
    dt2 = np.abs(old_time[t_2nd] - t_loc)
    # this is the scaling between the two quaternions
    interpol = dt1 / (dt1 + dt2 + 1e-5)

    # now perform interpolation
    new_q = slerp(q1, q2, interpol)

    return np.transpose(new_q)


def slerp(q0, q1, interpol, thresh=0.9995) -> float:
    """
    Interpolate the quaternion states,
    using the slerp algorithm: https://goo.gl/1f41aG
    t : [0,1] 0 => match to q0 and q1 accordingly
    """
    dot = np.sum(q0 * q1)

    # catch the case that the dot product is negative: -> avoids long path!
    if (dot < 0.0):
        q0 = -q0
        dot = -dot

    # if close to one linear interpolate and normalize!
    if (dot > thresh):
        result = q0 + interpol * (q1 - q0)  # linear!
        result = result / np.linalg.norm(result)  # normalize
        return result

    # else do normal slerp interpolation:
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * interpol

    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0) + (s1 * q1)


def do_3d_sync(loc_rob, loc_tr, dims=3, sign="pos"):
    """
    Synchronize the signals via 3d arrangement!
    Not properly working, can be deleted
    """
    # definitions:
    but_ord = 4
    but_freq = 0.01
    ranging = 0.2

    # go over all selected dimensions
    k = 0
    # get the arrays
    tr_pos, rob_pos = arrange(loc_rob, loc_tr, pos_rel=k, sign=sign)
    # apply filter
    rob_pos_f = high_pass(
        rob_pos, but_ord=but_ord, but_freq=5 * but_freq)
    tr_pos_f = high_pass(
        tr_pos, but_ord=but_ord, but_freq=but_freq)

    # get interesting part
    buffer, position = get_remarkable_part(
        tr_pos_f, ranging=ranging)

    dist = np.zeros(len(range(len(buffer), len(rob_pos_f))))

    for k in range(dims):
        # get the arrays
        tr_pos, rob_pos = arrange(loc_rob, loc_tr, pos_rel=k, sign=sign)
        buffer = tr_pos[position[0]:position[1]]

        # apply filter
        rob_pos_f = high_pass(
            rob_pos, but_ord=but_ord, but_freq=5 * but_freq)
        tr_pos_f = high_pass(
            tr_pos, but_ord=but_ord, but_freq=but_freq)

        # get distances
        dist += match_robot(buffer, position, rob_pos_f)

    return np.argmin(dist) - position[0]
