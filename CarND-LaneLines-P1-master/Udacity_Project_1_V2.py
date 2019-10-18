import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import shutil, sys
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def region_of_interest(image, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(image)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    x_size = img.shape[1]
    y_size = img.shape[0]
    lines_slope_intercept = np.zeros(shape=(len(lines), 2))
    for index, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - x1 * slope
            lines_slope_intercept[index] = [slope, intercept]
    max_slope_line = lines_slope_intercept[lines_slope_intercept.argmax(axis=0)[0]]
    min_slope_line = lines_slope_intercept[lines_slope_intercept.argmin(axis=0)[0]]
    left_slopes = []
    left_intercepts = []
    right_slopes = []
    right_intercepts = []
    # this gets slopes and intercepts of lines similar to the lines with the max (immediate left) and min
    # (immediate right) slopes (i.e. slope and intercept within x%)
    for line in lines_slope_intercept:
        if abs(line[0] - max_slope_line[0]) < 0.15 and abs(line[1] - max_slope_line[1]) < (0.15 * x_size):
            left_slopes.append(line[0])
            left_intercepts.append(line[1])
        elif abs(line[0] - min_slope_line[0]) < 0.15 and abs(line[1] - min_slope_line[1]) < (0.15 * x_size):
            right_slopes.append(line[0])
            right_intercepts.append(line[1])
    # left and right lines are averages of these slopes and intercepts, extrapolate lines to edges and center*
    # *roughly
    new_lines = np.zeros(shape=(1, 2, 4), dtype=np.int32)
    if len(left_slopes) > 0:
        left_line = [sum(left_slopes) / len(left_slopes), sum(left_intercepts) / len(left_intercepts)]
        left_bottom_x = (y_size - left_line[1]) / left_line[0]
        left_top_x = (y_size * .575 - left_line[1]) / left_line[0]

        if (left_bottom_x >= 0):
            y_new = y_size * .575
            new_lines[0][0] = [left_bottom_x, y_size, left_top_x, y_new]
    if len(right_slopes) > 0:
        right_line = [sum(right_slopes) / len(right_slopes), sum(right_intercepts) / len(right_intercepts)]
        right_bottom_x = (y_size - right_line[1]) / right_line[0]
        right_top_x = (y_size * .575 - right_line[1]) / right_line[0]
        if (right_bottom_x <= x_size):
            y_new = y_size * .575
            new_lines[0][1] = [right_bottom_x, y_size, right_top_x, y_new]
    for line in new_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_image = np.zeros((image.shape[0],image.shape[1],3), dtype=np.uint8)
    draw_lines(line_image,lines)
    return line_image

def weighted_img(image, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, image, β, γ)


# Python 3 has support for cool math symbols.
path = "test_images/"
test_imgs = os.listdir(path)
for img in test_imgs:
    lane_image = mpimg.imread("test_images/" + img)
    image = np.copy(lane_image)

    # Read image
    fig = plt.figure()
    fig.set_figheight(16)
    fig.set_figheight(12)

    # show image
    plot_1 = fig.add_subplot(4, 2, 1)
    plot_1.imshow(image)

    # convert in grayscale
    image_gray = grayscale(image)
    plot_2 = fig.add_subplot(4, 2, 2)
    plot_2.imshow(image_gray)

    # convert in HSV
    image_hsv = hsv(image)
    lower_yel = np.array([20,100,100])
    upper_yel = np.array([30,255,255])
    lower_wht = np.array([0,0,235])
    upper_wht = np.array([255,255,255])

    yellow_mask = cv2.inRange(image_hsv, lower_yel, upper_yel)
    white_mask = cv2.inRange(image_hsv, lower_wht, upper_wht)

    full_mask = cv2.bitwise_or(yellow_mask, white_mask)
    subdued_gray = (image_gray / 2).astype('uint8')
    boosted_lanes = cv2.bitwise_or(subdued_gray, full_mask)

    plot_3 = fig.add_subplot(4,2,3)
    plot_3.imshow(boosted_lanes)

    # apply gussian blur
    image_blur = gaussian_blur(boosted_lanes, 5)
    plot_4 = fig.add_subplot(4, 2, 4)
    plot_4.imshow(image_blur)

    # apply canny edge detection
    image_canny = canny(image_blur, 60, 150)
    plot_5 = fig.add_subplot(4, 2, 5)
    plot_5.imshow(image_canny)

    # apply region of interest
    x = image_canny.shape[1]
    y = image_canny.shape[0]
    vertices = np.array([[(x*0.,y),(x*.475, y*.575), (x*.525, y*.575), (x,y)]], dtype=np.int32)
    cropped_image = region_of_interest(image_canny, vertices)
    plot_6 = fig.add_subplot(4, 2, 6)
    plot_6.imshow(cropped_image)

    # apply Hough Transformation
    rho = 3
    theta = np.pi / 180
    threshold = 70
    min_line_len = 70
    max_line_gap = 250
    lines = hough_lines(cropped_image, rho, theta, threshold, min_line_len, max_line_gap)

    # display lines on original image
    combo_image = cv2.addWeighted(image, 0.8, lines, 1, 1)
    plot_7 = fig.add_subplot(4, 2, 7)
    plot_6.imshow(combo_image)

def finding_lane_pipline(image):
    #convert in grayscale
    image_gray = grayscale(image)

    #convert in HSV
    image_hsv = hsv(image)

    lower_yel = np.array([20,100,100])
    upper_yel = np.array([30,255,255])
    lower_wht = np.array([0,0,235])
    upper_wht = np.array([255,255,255])

    yellow_mask = cv2.inRange(image_hsv, lower_yel, upper_yel)
    white_mask = cv2.inRange(image_hsv, lower_wht, upper_wht)

    full_mask = cv2.bitwise_or(yellow_mask, white_mask)
    subdued_gray = (image_gray / 2).astype('uint8')
    boosted_lanes = cv2.bitwise_or(subdued_gray, full_mask)

    # apply gussian blur
    image_blur = gaussian_blur(boosted_lanes, 5)

    #apply canny edge detection
    image_canny = canny(image_blur, 60, 150)

    # apply region of interest
    x = image_canny.shape[1]
    y = image_canny.shape[0]
    vertices = np.array([[(x*0.,y),(x*.475, y*.575), (x*.525, y*.575), (x,y)]], dtype=np.int32)
    cropped_image = region_of_interest(image_canny, vertices)

    # apply Hough Transformation
    rho = 3
    theta = np.pi / 180
    threshold = 70
    min_line_len = 70
    max_line_gap = 250
    lines = hough_lines(cropped_image, rho, theta, threshold, min_line_len, max_line_gap)

    # display lines on original image
    result = cv2.addWeighted(image, 0.8, lines, 1, 1)
    return result


white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(finding_lane_pipline)  # NOTE: this function expects color images!!
# %time white_clip.write_videofile(white_output, audio=False)
white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(finding_lane_pipline)
# %time yellow_clip.write_videofile(yellow_output, audio=False)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(finding_lane_pipline)
# %time challenge_clip.write_videofile(challenge_output, audio=False)
challenge_clip.write_videofile(challenge_output, audio=False)


cv2.destroyAllWindows()


print("Success!!!")
