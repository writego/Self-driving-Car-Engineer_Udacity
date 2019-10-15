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

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def make_coordinates(image, line_parameters):
    slope,intercept = line_parameters
    # print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) # y=mx+b
    x1 = int((y1-intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    # print(left_fit)
    # print(right_fit)
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    # print(left_fit_average,'left')
    # print(right_fit_average,'right')
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])

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

def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines

def display_lines(image, lines, color=[255, 0, 0], thickness=2):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def weighted_img(image, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, image, β, γ)


# Python 3 has support for cool math symbols.
path = "test_images/"
test_imgs = os.listdir(path)
for img in test_imgs:
    lane_image = mpimg.imread("test_images/"+img)
    image = np.copy(lane_image)

    # Read image
    fig = plt.figure()
    fig.set_figheight(16)
    fig.set_figheight(12)

    #show image
    plot_1 = fig.add_subplot(4,2,1)
    plot_1.imshow(image)

    #convert in grayscale
    image_gray = grayscale(image)
    plot_2 = fig.add_subplot(4,2,2)
    plot_2.imshow(image_gray)

    #apply gussian blur
    image_blur = gaussian_blur(image_gray,5)
    plot_3 = fig.add_subplot(4,2,3)
    plot_3.imshow(image_blur)

    #apply canny edge detection
    image_canny = canny(image_blur,50,150)
    plot_4 = fig.add_subplot(4,2,4)
    plot_4.imshow(image_canny)

    #apply region of interest
    vertices = np.array([[(0, image.shape[0]), (465, 320), (475, 320), (image.shape[1], image.shape[0])]], dtype=np.int32)
    cropped_image = region_of_interest(image_canny,vertices)
    plot_4 = fig.add_subplot(4,2,4)
    plot_4.imshow(cropped_image)

    #apply Hough Transformation
    rho = 2
    theta = np.pi/180
    threshold = 45
    min_line_len = 40
    max_line_gap = 100
    lines = hough_lines(cropped_image, rho, theta, threshold, min_line_len, max_line_gap)

    #display lines on black screen
    line_image =display_lines(image,lines)
    plot_5 = fig.add_subplot(4,2,5)
    plot_5.imshow(line_image)

    #display lines on original image
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    plot_6 = fig.add_subplot(4,2,6)
    plot_6.imshow(combo_image)

    # optimizing
    averaged_lines = average_slope_intercept(image, lines)
    line_image = display_lines(image, averaged_lines)
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    plot_7 = fig.add_subplot(4, 2, 7)
    plot_7.imshow(combo_image)

def finding_lane_pipline(image):
    gray = grayscale(image)
    blur = gaussian_blur(gray,5)
    edges = canny(blur,50,150)
    vertices = np.array([[(0, image.shape[0]), (465, 320), (475, 320), (image.shape[1], image.shape[0])]], dtype=np.int32)
    cropped_image = region_of_interest(edges,vertices)
    lines = hough_lines(cropped_image, 2, np.pi/180, 45, 40, 100)
    averaged_lines = average_slope_intercept(image, lines)
    line_image = display_lines(image, averaged_lines)
    result = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return result


white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(finding_lane_pipline) #NOTE: this function expects color images!!
# %time white_clip.write_videofile(white_output, audio=False)
white_clip.write_videofile(white_output,audio=False)

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


print("Success")
