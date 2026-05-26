import cv2
import os
import numpy as np
from rect_proposal import rect_proposal_foreground, rect_consistency_edge, rect_consistency_of
import math
import matplotlib.pyplot as plt

# Choose video style
type_video = 3  # 1: short video; 2: long video; 3: two-phase multi-slug; 4: three-phase multi-slug.
index_video = 2

if type_video == 1:
    if index_video == 1:
        filename_all = '002127S5.avi'  # short video
    elif index_video == 2:
        filename_all = '002127S8.avi'
    elif index_video == 3:
        filename_all = '002127S10.avi'
    else:
        print('Invalid input')
        exit()
    n_frame_start = 100
    n_frame_end = 500
elif type_video == 2:
    filename_all = '002118s2-3rd.mp4'  # long video
    n_frame_start = 490
    n_frame_end = 550
elif type_video == 3:
    if index_video == 1:
        filename_all = 'DSC_0268_After1.mov'  # two-phase multi-slug
    elif index_video == 2:
        filename_all = 'DSC_0269_Before1.mov'
    else:
        print('Invalid input')
        exit()
    n_frame_start = 20
    n_frame_end = 300
elif type_video == 4:
    if index_video == 1:
        filename_all = '008021F2_AFTER.mov'  # three-phase multi-slug
    elif index_video == 2:
        filename_all = '008021F4_AFTER.mov'
    elif index_video == 3:
        filename_all = '008025S3_AFTER.mov'
    else:
        print('Invalid input')
        exit()
    n_frame_start = 20
    n_frame_end = 100
else:
    print('Invalid input')
    exit()

# Parameters
k_size_1 = 7  # opening parameters
k_size_2 = 5  # closing parameters
kernel1 = np.ones((k_size_1, k_size_1), np.uint8)
kernel2 = np.ones((k_size_2, k_size_2), np.uint8)

filename, _ = os.path.splitext(filename_all)
textname = filename + '_slug.txt'
fh = open(textname, 'w')
fh.write("Frame, Slug, Height, CenterLen, TopLen, BottomLen, EndLoc\n")

# Extract basic information
cap = cv2.VideoCapture(filename_all)

# Foreground detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=12)
list_up_contour = []
list_low_contour = []
rect_refined = []
list_contour = []
list_upper_points = []
list_lower_points = []
list_upper_points2 = []
list_lower_points2 = []
list_front_points = []
list_back_points = []
list_speed = []
list_speed_of = []
list_consist_edge = []
list_consist_of = []

peak_x = []
b_dir_sum = 0
preFrame = 0

currentFrame = 0
slurry_index = 0
n_single_slug = 0
w_dist = 10
b_fit = 1  # whether to fit the tube wall

while 1:
    ret, frame = cap.read()
    if not ret:
        break

    currentFrame = currentFrame + 1
    if currentFrame == n_frame_end:
        break

    if currentFrame < n_frame_start - 10:
        continue

    fgmask = fgbg.apply(frame)
    fgmask_or = fgmask.copy()
    image_name = f'_f{currentFrame}.jpg'

    if currentFrame == n_frame_start:
        bimask = fgmask.copy()
        bimask_sob = bimask.copy()
        mask = np.zeros_like(frame)
        mask[..., 1] = 255
        frame_pre = frame
        gray_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)

        gray_blur = cv2.GaussianBlur(gray_pre, (5, 5), 0)
        sobelmask_pre = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=5)

    if n_frame_start < currentFrame < n_frame_end:

        # Mixture-Gaussian model
        # fgmask = cv2.medianBlur(fgmask, 7)
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)

        bimask[fgmask > 60] = 1
        bimask[fgmask <= 60] = 0
        bimask = cv2.morphologyEx(bimask, cv2.MORPH_CLOSE, kernel2)
        bimask = cv2.morphologyEx(bimask, cv2.MORPH_OPEN, kernel1)

        bimask2 = bimask.copy()

        # Sobel edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        sobelmask = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobelmask_diff = abs(sobelmask) - abs(sobelmask_pre)
        # print(sobelmask.min())
        sobelmask_diff = (sobelmask_diff > 0.03 * abs(sobelmask).max()).astype(int)
        sobelmask_diff = sobelmask * sobelmask_diff

        # Otss' binarization
        # ret, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Opticalflow
        flow = cv2.calcOpticalFlowFarneback(gray_pre, gray,
                                            None,
                                            0.5, 3, 15, 3, 5, 1.1, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # mask[..., 0] = angle * 180 / np.pi / 2
        # mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # flow_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        frame_pre = frame
        gray_pre = gray
        sobelmask_pre = sobelmask
        # print([currentFrame, np.sum(bimask)])
        if np.sum(bimask) > 100:

            # Foreground based analysis
            rect_proposed = rect_proposal_foreground(bimask, filename, currentFrame)
            if not rect_proposed:
                continue

            frame_new = frame.copy()
            frame_new_c = frame.copy()
            frame_pro = frame.copy()
            frame_new2 = frame.copy()
            frame_res = frame.copy()
            analysis = cv2.connectedComponentsWithStats(bimask, 4, cv2.CV_32S)
            (totalLabels, label_ids, values, centroid) = analysis

            i_region = 0
            bimask2 = 0 * bimask2
            rect_refined.clear()
            list_contour.clear()

            # Refine proposals
            for [y_low, y_high, x_left, x_right] in rect_proposed:

                pt1 = (x_left, y_low)
                pt2 = (x_right, y_high)
                cv2.rectangle(frame_pro, pt1, pt2, (255, 255, 0), 3)

                b_refined = 0
                i_region = i_region + 1
                area_sum = 0

                for i in range(1, totalLabels):
                    area = values[i, cv2.CC_STAT_AREA]

                    if area > 50:
                        componentMask = (label_ids == i).astype("uint8")
                        inter_area = np.sum(componentMask[y_low:y_high, x_left:x_right])
                        if inter_area > 20:
                            componentMask = componentMask * 255
                            contour_each, _ = cv2.findContours(componentMask, cv2.RETR_EXTERNAL,
                                                               cv2.CHAIN_APPROX_TC89_L1)  # Original: CHAIN_APPROX_NONE
                            contour_each = np.squeeze(contour_each[0], 1)
                            area_sum += area
                            # cv2.polylines(frame_new, [contour_each], True, (0, 255, 0), 3)

                            bimask2[label_ids == i] = i_region

                            x1 = values[i, cv2.CC_STAT_LEFT]
                            y1 = values[i, cv2.CC_STAT_TOP]
                            w = values[i, cv2.CC_STAT_WIDTH]
                            h = values[i, cv2.CC_STAT_HEIGHT]

                            if b_refined == 0:
                                b_refined = 1
                                y_low2 = y1
                                y_high2 = y1 + h
                                x_left2 = x1
                                x_right2 = x1 + w
                                contour_region = contour_each
                            else:
                                if y_low2 > y1:
                                    y_low2 = y1
                                if y_high2 < y1 + h:
                                    y_high2 = y1 + h
                                if x_left2 > x1:
                                    x_left2 = x1
                                if x_right2 < x1 + w:
                                    x_right2 = x1 + w
                                contour_region = np.concatenate((contour_region, contour_each), axis=0)
                if area_sum > 150:
                    rect_refined.append([y_low2, y_high2, x_left2, x_right2])
                    list_contour.append(contour_region)

            # if len(rect_refined) > 5:
            #     list_up_contour.clear()
            #     list_low_contour.clear()
            if len(rect_refined) < 2:
                continue

            if abs(n_single_slug) <= 6:
                if len(rect_refined) == 2:
                    n_single_slug += 1
                else:
                    n_single_slug -= 1

            # Check the type of the video
            if n_single_slug > 5 and len(rect_refined) >= 3:
                continue
            if n_single_slug < -5 and len(rect_refined) < 3:
                continue

            # Edge based analysis
            sobelmask_x = np.sum(abs(sobelmask_diff), axis=0)
            edge_right = 0
            edge_left = 0
            # sobelmask_y = np.sum(sobelmask_diff, axis=1)

            x_location = np.zeros(len(rect_refined))
            for i_region in range(len(rect_refined)):

                [y_low, y_high, x_left, x_right] = rect_refined[i_region]
                x_location[i_region] = x_left
                y_margin = 0.08 * (y_high - y_low)
                contour_each = list_contour[i_region]
                if len(list_up_contour) > 10:
                    list_up_contour.pop(0)
                if len(list_low_contour) > 10:
                    list_low_contour.pop(0)
                index_up = contour_each[:, 1] > y_high - y_margin
                index_low = contour_each[:, 1] < y_low + y_margin

                list_up_contour.append(contour_each[index_up, :])
                list_low_contour.append(contour_each[index_low, :])

                if abs(b_dir_sum) < 5:
                    edge_left += sum(sobelmask_x[x_left:int((x_left + x_right) / 2.0)])
                    edge_right += edge_right + sum(sobelmask_x[int((x_left + x_right) / 2.0):x_right])

            # Average distance of those slugs
            if len(rect_refined) >= 3:
                x_dis = np.mean(x_location[1:len(x_location)] - x_location[0:len(x_location) - 1])
                if x_dis < 20:
                    continue

            up_contour_all = np.concatenate(list_up_contour, axis=0)
            low_contour_all = np.concatenate(list_low_contour, axis=0)

            if len(up_contour_all) < 15 or len(low_contour_all) < 15:
                list_up_contour.clear()
                list_low_contour.clear()
                continue

            # Determine direction
            if abs(b_dir_sum) < 5:
                if edge_left > edge_right:
                    b_dir_now = -1
                else:
                    b_dir_now = 1
                b_dir_sum = b_dir_sum + b_dir_now
            b_dir = np.sign(b_dir_sum)
            if b_dir == 0:
                continue

            vx1, vy1, cx1, cy1 = cv2.fitLine(up_contour_all, cv2.DIST_L1, 0, 0.01, 0.01)
            vx2, vy2, cx2, cy2 = cv2.fitLine(low_contour_all, cv2.DIST_L1, 0, 0.01, 0.01)

            w1 = (cx1 - 50) / vx1
            w2 = (1870 - cx1) / vx1
            cv2.line(frame_new2, (int(cx1 - vx1 * w1), int(cy1 - vy1 * w1)), (int(cx1 + vx1 * w2), int(cy1 + vy1 * w2)),
                     (200, 0, 200), 2)
            w1 = (cx2 - 50) / vx2
            w2 = (1870 - cx2) / vx2
            cv2.line(frame_new2, (int(cx2 - vx2 * w1), int(cy2 - vy2 * w1)), (int(cx2 + vx2 * w2), int(cy2 + vy2 * w2)),
                     (200, 0, 200), 2)

            list_upper_points.clear()
            list_lower_points.clear()
            list_upper_points.clear()
            list_lower_points2.clear()
            list_upper_points2.clear()
            list_front_points.clear()
            list_back_points.clear()

            b_type = np.ones(len(rect_refined))
            cri_front_q = np.ones(len(rect_refined))
            cri_front_c = np.ones(len(rect_refined))
            cri_front_b = np.ones(len(rect_refined))

            for i_region in range(len(rect_refined)):

                [y_low, y_high, x_left, x_right] = rect_refined[i_region]
                contour_each = list_contour[i_region]

                # Find types of two detections
                if len(rect_refined) < 3:
                    if b_dir == -1:
                        b_type[i_region] = i_region
                    else:
                        b_type[i_region] = 1 - i_region
                else:
                    if i_region == 0:
                        b_type[i_region] = 0
                    else:
                        x_left_pre = rect_refined[i_region - 1][2]
                        if x_left - x_left_pre < 2.1 * x_dis:
                            b_type[i_region] = (b_type[i_region - 1] + 1) % 2
                        else:
                            int_gap = math.floor(float(x_left - x_left_pre) / x_dis)
                            b_type[i_region] = (b_type[i_region - 1] + int_gap) % 2

                # Find front points
                bimask_now = bimask2[y_low:y_high, x_left:x_right]
                bimask_now = (bimask_now == i_region + 1).astype(int)

                y_idx = np.asarray(np.nonzero(np.sum(bimask_now, axis=1))[0])
                x_idx = np.zeros(len(y_idx))
                x_idx2 = np.zeros(len(y_idx))
                for j in range(len(y_idx)):
                    if b_dir == 1:
                        x_idx[j] = np.max(np.nonzero(bimask_now[y_idx[j],])[0])
                        x_idx2[j] = np.min(np.nonzero(bimask_now[y_idx[j],])[0])
                    else:
                        x_idx[j] = np.min(np.nonzero(bimask_now[y_idx[j],])[0])
                        x_idx2[j] = np.max(np.nonzero(bimask_now[y_idx[j],])[0])

                y_idx = (y_idx + y_low).astype('int32')
                x_idx = (x_idx + x_left).astype('int32')
                front_points = np.vstack((x_idx, y_idx)).transpose()
                back_points = np.vstack((x_idx2, y_idx)).transpose()

                if len(front_points) > 5:
                    p = np.polyfit(front_points[:, 1], front_points[:, 0], 2)
                    cri_front_q[i_region] = p[0]
                else:
                    cri_front_q[i_region] = 0

                if np.sum((bimask2 == i_region + 1).astype(int)) > 1:
                    cri_front_c[i_region] = np.std(gray[bimask2 == i_region + 1])
                    blue_temp = frame[bimask2 == i_region + 1, 0]
                    cri_front_b[i_region] = np.mean(blue_temp / gray[bimask2 == i_region + 1])
                else:
                    cri_front_c[i_region] = 0
                    cri_front_b[i_region] = 0

                list_front_points.append(front_points)
                list_back_points.append(back_points)
                cv2.polylines(frame_new2, [front_points], False, (0, 255, 0), 3)
                # print(front_points)

                dis_upper = abs(vy1 * (contour_each[:, 0] - cx1) - vx1 * (contour_each[:, 1] - cy1))
                dis_upper = dis_upper - 3
                dis_upper[dis_upper < 0] = 0
                # Without fitting
                if b_fit == 0:
                    dis_upper = -contour_each[:, 1]

                dis_lower = abs(vy2 * (contour_each[:, 0] - cx2) - vx2 * (contour_each[:, 1] - cy2))
                dis_lower = dis_lower - 3
                dis_lower[dis_lower < 0] = 0
                # Without fitting
                if b_fit == 0:
                    dis_lower = contour_each[:, 1]

                if b_dir == 1:
                    cri_upper = contour_each[:, 0] - w_dist * dis_upper
                    index_upper = np.argmax(cri_upper)
                    cri_lower = contour_each[:, 0] - w_dist * dis_lower
                    index_lower = np.argmax(cri_lower)

                    cri_upper = contour_each[:, 0] + w_dist * dis_upper
                    index_upper2 = np.argmin(cri_upper)
                    cri_lower = contour_each[:, 0] + w_dist * dis_lower
                    index_lower2 = np.argmin(cri_lower)
                else:
                    cri_upper = contour_each[:, 0] + w_dist * dis_upper
                    index_upper = np.argmin(cri_upper)
                    cri_lower = contour_each[:, 0] + w_dist * dis_lower
                    index_lower = np.argmin(cri_lower)

                    cri_upper = contour_each[:, 0] - w_dist * dis_upper
                    index_upper2 = np.argmax(cri_upper)
                    cri_lower = contour_each[:, 0] - w_dist * dis_lower
                    index_lower2 = np.argmax(cri_lower)

                point_upper = contour_each[index_upper, :]
                point_lower = contour_each[index_lower, :]
                cv2.circle(frame_new2, (point_upper[0], point_upper[1]), 10, (0, 0, 255), -1)
                cv2.circle(frame_new2, (point_lower[0], point_lower[1]), 10, (0, 0, 255), -1)
                list_upper_points.append(point_upper)
                list_lower_points.append(point_lower)

                point_upper = contour_each[index_upper2, :]
                point_lower = contour_each[index_lower2, :]
                # cv2.circle(frame_new2, (point_upper[0], point_upper[1]), 10, (255, 0, 0), -1)
                # cv2.circle(frame_new2, (point_lower[0], point_lower[1]), 10, (255, 0, 0), -1)
                list_upper_points2.append(point_upper)
                list_lower_points2.append(point_lower)

            if sum((b_type == 0).astype(int)) < 1 or sum((b_type == 1).astype(int)) < 0:
                continue

            if len(rect_refined) >= 3:
                cri_q_0 = np.mean(abs(cri_front_q[b_type == 0]))
                cri_q_1 = np.mean(abs(cri_front_q[b_type == 1]))
                cri_i_0 = np.mean(cri_front_c[b_type == 0])
                cri_i_1 = np.mean(cri_front_c[b_type == 1])
                cri_b_0 = np.mean(cri_front_b[b_type == 0])
                cri_b_1 = np.mean(cri_front_b[b_type == 1])
                if max(cri_q_0, cri_q_1) / (min(cri_q_0, cri_q_1) + 1e-8) < 5 or max(cri_b_0, cri_b_1) > 1.05:
                    if cri_i_0 > cri_i_1:
                        b_type = 1 - b_type
                else:
                    if cri_q_0 < cri_q_1:
                        b_type = 1 - b_type
            # print([cri_q_0, cri_q_1, cri_i_0, cri_i_1])

            b_type_pre = 0
            if len(rect_refined) >= 3:
                slurry_index = 0

            peak_x_pre = peak_x.copy()
            peak_x.clear()
            for i_region in range(len(rect_refined)):
                if b_dir == 1:
                    i_region_now = i_region
                else:
                    i_region_now = len(rect_refined) - 1 - i_region

                [y_low, y_high, x_left, x_right] = rect_refined[i_region_now]

                # Update slurry index for single slurry
                if i_region == 0 and len(rect_refined) < 3:
                    if slurry_index == 0:
                        slurry_index = 1
                    else:
                        if b_dir * x_left_pre > b_dir * x_left:
                            slurry_index += 1
                    x_left_pre = x_left

                pt1 = (x_left, y_low)
                pt2 = (x_right, y_high)

                cv2.rectangle(frame_new, pt1, pt2, (255, 255, 0), 3)
                if b_type[i_region_now] == 0:
                    cv2.rectangle(frame_new_c, pt1, pt2, (255, 255, 0), 3)
                else:
                    cv2.rectangle(frame_new_c, pt1, pt2, (0, 255, 255), 3)

                # Six point models
                if b_type[i_region_now] == 1:
                    point_2 = list_lower_points[i_region_now]
                    point_3 = list_upper_points[i_region_now]

                    point_2_end = list_lower_points2[i_region_now]
                    point_3_end = list_upper_points2[i_region_now]

                    dis_2 = abs(point_2[0] - point_2_end[0])
                    if dis_2 < 1:
                        dis_2 = 1
                    dis_3 = abs(point_3[0] - point_3_end[0])
                    if dis_3 < 1:
                        dis_3 = 1

                    if dis_2 > dis_3 + 5 and len(rect_refined) < 3 and b_fit == 1:
                        point_2[0] = point_2_end[0] + point_3[0] - point_3_end[0]
                        point_2[1] = point_2_end[1] + (point_2[1] - point_2_end[1]) * dis_3 / dis_2
                    if dis_3 > dis_2 + 5 and len(rect_refined) < 3 and b_fit == 1:
                        point_3[0] = point_3_end[0] + point_2[0] - point_2_end[0]
                        point_3[1] = point_3_end[1] + (point_3[1] - point_3_end[1]) * dis_2 / dis_3

                    point_s = [int((point_2[0] + point_3[0]) / 2.0), int((point_2[1] + point_3[1]) / 2.0)]

                    b_type_pre = 1

                if b_type[i_region_now] == 0 and b_type_pre == 1:
                    point_1 = list_lower_points[i_region_now]
                    point_4 = list_upper_points[i_region_now]
                    if len(rect_refined) >= 3:
                        slurry_index = slurry_index + 1
                    b_type_pre = 0

                    pts = np.array([point_1, point_2, point_3, point_4], np.int32)
                    pts = pts.reshape((-1, 1, 2))

                    front_contour = list_front_points[i_region_now]
                    if len(front_contour) < 5:
                        continue
                    p = np.poly1d(np.polyfit(front_contour[:, 1], front_contour[:, 0], 3))
                    point_e = [int(p((point_4[1] + point_1[1]) / 2.0)), int((point_4[1] + point_1[1]) / 2.0)]

                    peak_x.append((point_2[0] + point_3[0]) / 2.0)
                    peak_x.append((point_1[0] + point_4[0]) / 2.0)

                    cv2.polylines(frame_res, [pts], True, (255, 255, 0), 2)
                    cv2.line(frame_res, tuple(point_e), tuple(point_s), (255, 255, 0), 2)

                    slug_height = (point_4[1] - point_1[1] + point_3[1] - point_2[1]) * 0.5
                    slug_top = np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)
                    slug_bottom = np.sqrt((point_3[0] - point_4[0]) ** 2 + (point_3[1] - point_4[1]) ** 2)
                    slug_center = np.sqrt((point_e[0] - point_s[0]) ** 2 + (point_e[1] - point_s[1]) ** 2)

                    line = np.array(
                        [currentFrame, slurry_index, slug_height, slug_center, slug_top, slug_bottom, point_s[0]])
                    np.savetxt(fh, [line], fmt='%.2f', delimiter=',')

            # bimask2 = cv2.normalize(bimask2, None, 0, 255, cv2.NORM_MINMAX)
            # image_name = f'_f{currentFrame}_region.jpg'
            # cv2.imwrite('data/' + filename + image_name, bimask2)

            # sobelmask_x = np.sum(sobelmask_diff, axis=0)
            # sobelmask_y = np.sum(sobelmask_diff, axis=1)
            # plt.figure()
            # plt.subplot(211)
            # plt.plot(sobelmask_x)
            # plt.subplot(212)
            # plt.plot(sobelmask_y)
            # image_name = f'_f{currentFrame}_sum.png'
            # plt.savefig('data/' + filename + image_name)
            # plt.close()
            error_rate = rect_consistency_edge(sobelmask_diff, peak_x, b_dir, filename, currentFrame)
            speed_slug = 0

            if len(peak_x_pre) > 1 and len(peak_x) > 1 and preFrame == currentFrame - 1:
                speed_slug_all = []
                peak_x_pre = np.array(peak_x_pre)
                for peak in peak_x:
                    speed_slug_all.append(np.min(abs(peak_x_pre - peak)))
                speed_slug = np.median(np.array(speed_slug_all))
                if speed_slug > 200:
                    speed_slug = 0

            preFrame = currentFrame
            print('Curent frame: ' + filename + image_name)
            # print(b_dir)
            if speed_slug > 0:
                list_speed.append(b_dir * speed_slug)
                print(f'The speed of the slug (left to right): {b_dir * speed_slug:.2f}')

            list_consist_edge.append(error_rate)
            con_cri = error_rate
            str_con_cri = '{:.4f}'.format(con_cri)
            print(f'The consistency score based on the Sobel edge {error_rate:.4f}.')

            error_rate, speed_slug_op = rect_consistency_of(magnitude, angle, peak_x, b_dir, filename, currentFrame)
            list_consist_of.append(error_rate)
            print(f'The consistency score based on the optical flow {error_rate:.4f}.')

            if speed_slug > 0 and speed_slug_op > 0:
                list_speed_of.append(b_dir * speed_slug_op)
                print(f'The speed of the slug from the optical flow (left to right): {b_dir * speed_slug_op:.2f}')
            print('\n')

            if con_cri < 0.75:
                # cv2.imwrite('data/' + filename + image_name, frame)

                # image_name = f'_f{currentFrame}_fg.jpg'
                # cv2.imwrite('data/' + filename + image_name, fgmask_or)

                # image_name = f'_f{currentFrame}_otsu.jpg'
                # cv2.imwrite('data/' + filename + image_name, th)

                # image_name = f'_f{currentFrame}_edge.jpg'
                # cv2.imwrite('data/' + filename + image_name, abs(sobelmask_diff) * 10)

                # image_name = f'_f{currentFrame}_edge_or.jpg'
                # cv2.imwrite('data/' + filename + image_name, abs(sobelmask) * 10)

                # image_name = f'_f{currentFrame}_flow.jpg'
                # cv2.imwrite('data/' + filename + image_name, flow_rgb)

                # image_name = f'_f{currentFrame}_pro.jpg'
                # cv2.imwrite('data/' + filename + image_name, frame_pro)
                #
                # image_name = f'_f{currentFrame}_bimask.jpg'
                # cv2.imwrite('data/' + filename + image_name, (bimask2 > 0) * 255)
                #
                # image_name = f'_f{currentFrame}_region.jpg'
                # cv2.imwrite('data/' + filename + image_name, cv2.normalize(bimask2, None, 0, 255, cv2.NORM_MINMAX))
                #
                # image_name = f'_f{currentFrame}_pro_updated.jpg'
                # cv2.imwrite('data/' + filename + image_name, frame_new)
                #
                # image_name = f'_f{currentFrame}_pro_updated_c.jpg'
                # cv2.imwrite('data/' + filename + image_name, frame_new_c)
                #
                # image_name = f'_f{currentFrame}_contour.jpg'
                # cv2.imwrite('data/' + filename + image_name, frame_new2)

                if b_fit == 0:
                    image_name = f'_f{currentFrame}_slug_free.jpg'
                    cv2.imwrite('data/' + filename + image_name, frame_res)
                else:
                    image_name = f'_f{currentFrame}_slug_{str_con_cri}.jpg'
                    cv2.imwrite('data/' + filename + image_name, frame_res)
                # print(peak_x)
                # break

print(f'The average speed of the slug (left to right): {sum(list_speed) / len(list_speed):.2f}')
print(
    f'The average speed of the slug from the optical flow (left to right): {sum(list_speed_of) / len(list_speed_of):.2f}')
print(f'The average consistency score based on the Sobel edge {sum(list_consist_edge) / len(list_consist_edge):.4f}.')
print(f'The average consistency score based on the optical flow {sum(list_consist_of) / len(list_consist_of):.4f}.')

fh.close()
cap.release()
exit()
