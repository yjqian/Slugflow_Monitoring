import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from scipy.signal import find_peaks, medfilt


# Use foreground to general proposal
def rect_proposal_foreground(bimask, filename, currentFrame):

    fgmask_y = np.sum(bimask, axis=1)

    rect_proposed = []

    # Find y peaks
    p_tail = 0.2
    n_height = len(fgmask_y)
    fgmask_y[0:int(p_tail*n_height)] = 0
    fgmask_y[n_height-int(p_tail * n_height):n_height-1] = 0
    if fgmask_y.max() < 20:
        return rect_proposed
    fgmask_y = medfilt(fgmask_y, 51)

    xd = np.arange(0, n_height)
    xd = xd.reshape(n_height, 1)
    regr_y = DecisionTreeRegressor(max_leaf_nodes=3)
    regr_y.fit(xd, fgmask_y.reshape(n_height, 1))
    y_split = regr_y.tree_.threshold
    y_split = np.sort(y_split[y_split > 0])
    if len(y_split) < 2:
        return rect_proposed
    if y_split[1] - y_split[0] < 50:
        return rect_proposed
    y_low = int(y_split[0])
    y_high = int(y_split[1])

    bimask[0:int(y_split[0])-10, ] = 0
    bimask[int(y_split[1])+10:n_height-1, ] = 0
    fgmask_x = np.sum(bimask, axis=0)
    # fgmask_x = savgol_filter(fgmask_x, 21, 3)
    fgmask_x = np.convolve(fgmask_x, np.ones(21)/21, mode='valid')

    peaks_x, _ = find_peaks(fgmask_x, height=0.3 * np.max(fgmask_x), prominence=1, distance=60, width=10)
    if len(peaks_x) < 1:
        return rect_proposed

    n_width = len(fgmask_x)
    if len(peaks_x) > 1:
        w_x = np.percentile(peaks_x[1:len(peaks_x)] - peaks_x[0:len(peaks_x)-1], 10)
    else:
        w_x = 200

    for peak_now in peaks_x:
        x_left = int(peak_now - 0.25 * w_x)
        if x_left < 0:
            x_left = 0
        x_right = int(peak_now + 0.25 * w_x)
        if x_right > n_width - 1:
            x_right = n_width - 1

        rect_ratio = np.sum(bimask[y_low:y_high, x_left:x_right]) / (y_high - y_low) / (x_right - x_left)

        if rect_ratio > 0.01:
            rect_proposed.append([y_low, y_high, x_left, x_right])

    # plt.subplot(2, 1, 1)
    # plt.plot(peaks_x, fgmask_x[peaks_x], "ob")
    # plt.plot(fgmask_x)
    # plt.xlabel("Horizontal Image Locations")
    # plt.ylabel("Sum of Foreground Pixels")
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(fgmask_y)
    # plt.plot(xd, regr_y.predict(xd), 'r--')
    # plt.xlabel("Vertical Image Locations")
    # plt.ylabel("Sum of Foreground Pixels")
    # plt.tight_layout()
    #
    # image_name = f'_f{currentFrame}_sum.png'
    # plt.savefig('data/' + filename + image_name)
    # plt.close()

    return rect_proposed


def rect_consistency_edge(sobelmask_diff, peak_x, b_dir, filename, currentFrame):

    peak_x = np.array(peak_x)
    if b_dir == -1:
        peak_x = np.flip(peak_x)

    p_tail = 0.2
    n_height, _ = sobelmask_diff.shape
    edge_x = np.sum(sobelmask_diff[int(p_tail*n_height):n_height - int(p_tail * n_height), ], axis=0)
    edge_x = abs(edge_x)
    edge_x = np.convolve(edge_x, np.ones(21) / 21, mode='valid')

    # peak_x_edge1, _ = find_peaks(edge_x, height=0.3*np.max(abs(edge_x)), prominence=1, distance=60, width=10)
    # peak_x_edge2, _ = find_peaks(-edge_x, height=0.3 * np.max(abs(edge_x)), prominence=1, distance=60, width=10)
    # peak_x_edge = np.sort(np.concatenate([peak_x_edge1, peak_x_edge2]))
    peak_x_edge, _ = find_peaks(edge_x, height=0.3*np.max(edge_x), prominence=1, distance=60, width=10)

    n_peak = len(peak_x)
    if n_peak < 2 or len(peak_x_edge) < 1:
        return 0

    length_mean = np.average(peak_x[1:n_peak] - peak_x[0:n_peak-1])

    n_peak_edge = len(peak_x_edge)
    error_peak = 0
    for peak_fg in peak_x:
        error_peak += np.min(abs(peak_x_edge - peak_fg))
    if n_peak_edge > n_peak:
        error_peak += (n_peak_edge - n_peak) * length_mean
    error_peak = error_peak / max(n_peak, n_peak_edge)

    error_rate = 1 - error_peak/length_mean
    if error_rate < 0:
        error_rate = 0

    # print(length_mean)
    # print(error_peak)

    # plt.plot(edge_x, 'r-')
    # plt.plot(peak_x_edge, edge_x[peak_x_edge], "or")
    # plt.plot(peak_x, np.zeros(len(peak_x)), "Xb", markersize=15)
    # plt.xlabel("Horizontal Image Locations")
    # plt.ylabel("Sum of Sobel Edge Pixels")
    # image_name = f'_f{currentFrame}_sobel_peak.png'
    # plt.savefig('data/' + filename + image_name)
    # plt.close()

    return error_rate


def rect_consistency_of(magnitude, angle, peak_x, b_dir, filename, currentFrame):
    peak_x = np.array(peak_x)
    if b_dir == -1:
        peak_x = np.flip(peak_x)

    mag_mask = magnitude > 5
    # print(np.max(magnitude))
    angle = angle * 180 / np.pi
    dis_angle1 = np.minimum(angle, 360-angle)
    dis_angle2 = abs(180 - angle)
    angle_mask = (dis_angle1 < 15) | (dis_angle2 < 15)

    of_mask = (mag_mask & angle_mask)
    if np.sum(of_mask) < 50:
        return 0, 0

    speed = np.median(magnitude[of_mask])
    of_mask = of_mask.astype(int)

    p_tail = 0.2
    n_height, _ = of_mask.shape
    of_x = np.sum(of_mask[int(p_tail * n_height):n_height - int(p_tail * n_height), ], axis=0)
    of_x = np.convolve(of_x, np.ones(21) / 21, mode='valid')

    peak_x_of, _ = find_peaks(of_x, height=0.3 * np.max(of_x), prominence=1, distance=60, width=10)
    if len(peak_x_of) < 1:
        return 0, speed

    # plt.plot(of_x, 'r-')
    # plt.plot(peak_x_of, of_x[peak_x_of], "or")
    # plt.plot(peak_x, np.zeros(len(peak_x)), "Xb", markersize=15)
    # plt.xlabel("Horizontal Image Locations")
    # plt.ylabel("Sum of Optical Flow Pixels")
    # image_name = f'_f{currentFrame}_of_peak.png'
    # plt.savefig('data/' + filename + image_name)
    # plt.close()

    # image_name = f'_f{currentFrame}_opmask.jpg'
    # cv2.imwrite('data/' + filename + image_name, of_mask * 255)

    n_peak = len(peak_x)
    if n_peak < 2:
        return 0, speed

    length_mean = np.average(peak_x[1:n_peak] - peak_x[0:n_peak - 1])

    n_peak_of = len(peak_x_of)
    error_peak = 0
    for peak_fg in peak_x:
        error_peak += np.min(abs(peak_x_of - peak_fg))
    if n_peak_of > n_peak:
        error_peak += (n_peak_of - n_peak) * length_mean
    error_peak = error_peak / max(n_peak, n_peak_of)

    error_rate = 1 - error_peak / length_mean
    if error_rate < 0:
        error_rate = 0

    return error_rate, speed
