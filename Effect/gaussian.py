from scipy.ndimage import gaussian_filter1d
import cv2
import matplotlib.pyplot as plt
import numpy as np

def extract_coordinate_from_contours(contours):
    """
          This function seperate the array contour into two part. One part only contains x coordinate
          the other contains y coordinate
          :param
            --contours: the contour line
          :return
            x_coordinate: array of x coordinates
            y_coordinate: array of y coordinates
        """

    x_coordinate = []
    y_coordinate = []
    for cnt in contours:
        coordinate = cnt[0]
        x_coordinate.append(coordinate[1])
        y_coordinate.append(coordinate[0])

    return [x_coordinate, y_coordinate]


def show_line_with_diff_color(img, contours, color='r'):
    """This function is to hightlight to secific region of the line
            :param
                --contours: contour of the line
                --image: the image that want to motify
                --color: color of the secific region
            :return
                img: image with the highlight region in contour
    """

    # blank = np.zeros(img.shape, dtype=np.uint8)
    for cnt in contours:
        val = cnt[0]
        img[val[1], val[0], 1] = 255
        # blank[val[1], val[0], 1] = 255
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img


def smoothing_line(result_img, global_line_contours, local_line_contours, ranging, visualize, smoothenByX, smoothenByY, local_rate,
                   global_rate, long_rate, img_shape, highlight):
    """ This function emphasizes on smoothing specific region of the random line

          :param
          --global_line_contours: contours points of the whole line
          --local_line_contours: contours points of the regional line
          --ranging: range of contours of local line in global line
          --rate: how much user want to smoothen
          --smoothenByX: only smoothing by the X coordinate
          --smoothenByY: only smoothing by the Y coodinate
          --highlight: just show the local line with different color

         :return
            result_img: numpy image has been smoothened
            new_global_line: numpy contour has been smoothened
        """

    local_line_x, local_line_y = extract_coordinate_from_contours(local_line_contours)
    global_line_x, global_line_y = extract_coordinate_from_contours(global_line_contours)

    if highlight:
        cv2.drawContours(result_img, [global_line_contours], -1, (255, 0, 255), 1)

        result_img = show_line_with_diff_color(result_img, local_line_contours, 'r')

        return result_img, global_line_contours

    else:
        # Smoothing only for local line

        for i in range(1, local_rate):

            r_x = i if smoothenByX else 1
            r_y = i if smoothenByY else 1

            new_local_line_y = gaussian_filter1d(local_line_y, r_y)
            new_local_line_x = gaussian_filter1d(local_line_x, r_x)

            new_local_line = [[list(a)] for a in zip(new_local_line_y, new_local_line_x)]
            new_local_line = np.asarray(new_local_line)
            global_line_contours[ranging[0]:ranging[1]] = new_local_line

            # visuale lize the process local smoothing
            if visualize:
                replicate_global = global_line_contours.copy()

                blank = np.zeros(img_shape)
                blank = convert_color_img(blank, 'x')
                cv2.drawContours(blank, [global_line_contours], -1, (255, 0, 255), 1)
                blank = show_line_with_diff_color(blank, new_local_line, 'r')
                plt.imshow(blank)
                plt.show()

        # global gaussian
        global_line_x, global_line_y = extract_coordinate_from_contours(global_line_contours)

        # define
        # font, back meaning starting node and ending node respectively
        if global_rate != 0:
            font_start = ranging[0] - int(ranging[0] * long_rate)
            font_end = ranging[0] + int(ranging[0] * long_rate)
            back_start = ranging[1] - int(ranging[1] * long_rate)
            back_end = ranging[1] + int(ranging[1] * long_rate)

            # global smoothening

            global_line_x[font_start:font_end] = gaussian_filter1d(global_line_x[font_start:font_end], global_rate)
            global_line_y[font_start:font_end] = gaussian_filter1d(global_line_y[font_start:font_end], global_rate)
            global_line_x[back_start:back_end] = gaussian_filter1d(global_line_x[back_start:back_end], global_rate)
            global_line_y[back_start:back_end] = gaussian_filter1d(global_line_y[back_start:back_end], global_rate)
            global_line_y[font_start:back_end] = gaussian_filter1d(global_line_y[font_start:back_end], global_rate)

        new_global_line = [[list(a)] for a in zip(global_line_y, global_line_x)]
        new_global_line = np.asarray(new_global_line)

        result_img = cv2.drawContours(result_img, [new_global_line], -1, (255, 0, 255), 1)

        return result_img, new_global_line

def convert_color_img(img, color):
    """
    Convert color of character of binary image [0, 255]
    :param img: cv2 binary image
    :param color: 'r'/'b'/'g': convert to red/blue/green
    :return: numpy color image
    """
    cv_rgb_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    np_rgb_color = np.array(cv_rgb_img)
    noChannel = []
    if color == 'r':
        noChannel.append(0)
    elif color == 'g':
        noChannel.append(1)
    elif color == 'b':
        noChannel.append(2)
    else:
        noChanne = [0,1,2]
    for color_index in noChannel:
        np_rgb_color[np_rgb_color[:, :, color_index] == 0, color_index] = 255
    return np_rgb_colordef

def extract_coordinate_from_contours(contours):
    """
          This function seperate the array contour into two part. One part only contains x coordinate
          the other contains y coordinate
          :param
            --contours: the contour line
          :return
            x_coordinate: array of x coordinates
            y_coordinate: array of y coordinates
        """

    x_coordinate = []
    y_coordinate = []
    for cnt in contours:
        coordinate = cnt[0]
        x_coordinate.append(coordinate[1])
        y_coordinate.append(coordinate[0])

    return [x_coordinate, y_coordinate]


def show_line_with_diff_color(img, contours, color='r'):
    """This function is to hightlight to secific region of the line
            :param
                --contours: contour of the line
                --image: the image that want to motify
                --color: color of the secific region
            :return
                img: image with the highlight region in contour
    """

    # blank = np.zeros(img.shape, dtype=np.uint8)
    for cnt in contours:
        val = cnt[0]
        img[val[1], val[0], 1] = 255
        # blank[val[1], val[0], 1] = 255
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img


def smoothing_line(result_img, global_line_contours, local_line_contours, ranging, visualize, smoothenByX, smoothenByY, local_rate,
                   global_rate, long_rate, img_shape, highlight):
    """ This function emphasizes on smoothing specific region of the random line

          :param
          --global_line_contours: contours points of the whole line
          --local_line_contours: contours points of the regional line
          --ranging: range of contours of local line in global line
          --rate: how much user want to smoothen
          --smoothenByX: only smoothing by the X coordinate
          --smoothenByY: only smoothing by the Y coodinate
          --highlight: just show the local line with different color

         :return
            result_img: numpy image has been smoothened
            new_global_line: numpy contour has been smoothened
        """

    local_line_x, local_line_y = extract_coordinate_from_contours(local_line_contours)
    global_line_x, global_line_y = extract_coordinate_from_contours(global_line_contours)

    if highlight:
        cv2.drawContours(result_img, [global_line_contours], -1, (255, 0, 255), 1)

        result_img = show_line_with_diff_color(result_img, local_line_contours, 'r')

        return result_img, global_line_contours

    else:
        # Smoothing only for local line

        for i in range(1, local_rate):

            r_x = i if smoothenByX else 1
            r_y = i if smoothenByY else 1

            new_local_line_y = gaussian_filter1d(local_line_y, r_y)
            new_local_line_x = gaussian_filter1d(local_line_x, r_x)

            new_local_line = [[list(a)] for a in zip(new_local_line_y, new_local_line_x)]
            new_local_line = np.asarray(new_local_line)
            global_line_contours[ranging[0]:ranging[1]] = new_local_line

            # visuale lize the process local smoothing
            if visualize:
                replicate_global = global_line_contours.copy()

                blank = np.zeros(img_shape)
                blank = convert_color_img(blank, 'x')
                cv2.drawContours(blank, [global_line_contours], -1, (255, 0, 255), 1)
                blank = show_line_with_diff_color(blank, new_local_line, 'r')
                plt.imshow(blank)
                plt.show()

        # global gaussian
        global_line_x, global_line_y = extract_coordinate_from_contours(global_line_contours)

        # define
        # font, back meaning starting node and ending node respectively
        if global_rate != 0:
            font_start = ranging[0] - int(ranging[0] * long_rate)
            font_end = ranging[0] + int(ranging[0] * long_rate)
            back_start = ranging[1] - int(ranging[1] * long_rate)
            back_end = ranging[1] + int(ranging[1] * long_rate)

            # global smoothening

            global_line_x[font_start:font_end] = gaussian_filter1d(global_line_x[font_start:font_end], global_rate)
            global_line_y[font_start:font_end] = gaussian_filter1d(global_line_y[font_start:font_end], global_rate)
            global_line_x[back_start:back_end] = gaussian_filter1d(global_line_x[back_start:back_end], global_rate)
            global_line_y[back_start:back_end] = gaussian_filter1d(global_line_y[back_start:back_end], global_rate)
            global_line_y[font_start:back_end] = gaussian_filter1d(global_line_y[font_start:back_end], global_rate)

        new_global_line = [[list(a)] for a in zip(global_line_y, global_line_x)]
        new_global_line = np.asarray(new_global_line)

        result_img = cv2.drawContours(result_img, [new_global_line], -1, (255, 0, 255), 1)

        return result_img, new_global_line

def convert_color_img(img, color):
    """
    Convert color of character of binary image [0, 255]
    :param img: cv2 binary image
    :param color: 'r'/'b'/'g': convert to red/blue/green
    :return: numpy color image
    """
    cv_rgb_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    np_rgb_color = np.array(cv_rgb_img)
    noChannel = []
    if color == 'r':
        noChannel.append(0)
    elif color == 'g':
        noChannel.append(1)
    elif color == 'b':
        noChannel.append(2)
    else:
        noChanne = [0,1,2]
    for color_index in noChannel:
        np_rgb_color[np_rgb_color[:, :, color_index] == 0, color_index] = 255
    return np_rgb_color