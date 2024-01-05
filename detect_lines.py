import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_lines(img, sigma, threshold, numLines):
    """
    Detects lines in a color or grayscale image using the a custom implementation of the Hough transform.

    Parameters:
    img (numpy.ndarray): The input image.
    sigma (float): The standard deviation of the Gaussian blur applied to the image.
    threshold (int): The low threshold value used for Canny edge detection. The high threshold is twice this value.
    numLines (int): The maximum number of lines to detect.

    Returns:
    list: A list of tuples representing the detected lines. Each tuple contains
        two values: rho and theta, where rho is the distance from the origin to
        the line and theta is the angle between the x-axis and the normal to the line.
    """

    def blur_image(image, sigma_x, sigma_y):
        ksize = (int(2 * round(3 * sigma_x) + 1), int(2 * round(3 * sigma_y) + 1))
        return cv.GaussianBlur(image, ksize, sigma_x, sigmaY=sigma_y) 
       
    #Blur image
    blurred_img = blur_image(img, sigma, sigma)
    
    #Take Canny Edge detector
    edge_img = cv.Canny(blurred_img, threshold, 2 * threshold)

    #Set theta range to -pi/2 to pi/2 and calculate sin and cosine
    theta_range = np.deg2rad(np.arange(-90.0, 90.0))
    cos_theta, sin_theta = np.cos(theta_range), np.sin(theta_range)
    
    #Max distance = sqrt of height squared + width squared or the diagonal of the picture
    max_dist = int(np.sqrt(edge_img.shape[0]**2 + edge_img.shape[1]**2))
    
    #Create a matrix of zeros of size max distance by theta range
    accumulator = np.zeros((2 * max_dist, len(theta_range)), dtype=np.uint8)
    
    #Loop to check each pixel and cast a vote for which lines it is a part of and add vote to the accumulator matrix
    #If an edge is not at 255 intensity skip it
    #if it is at 255 add the sin and cosine and set to rho values
    #Add 1 to the accumulator for that pixel
    for y in range(len(edge_img)):
        for x in range(len(edge_img[0])):
            if edge_img[y, x] != 255:
                continue
            rho_values = x * cos_theta + y * sin_theta
            accumulator[np.round(rho_values + max_dist).astype(int), np.arange(len(theta_range))] += 1
    
    i = 0
    j = 0
    lines = []
    width, height = img.shape
    radius = 1.2
    R = int(radius)
    
    while i < numLines:
        rho_idx, theta_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        rho = rho_idx - max_dist
         
        #Attempt to 0 out neighborhood of chosen pizel (ran out of time)
        # row, column = rho_idx // width, rho_idx % width
        # x = np.arange(max(column - R, 0), min(column + R + 1, width))
        # y = np.arange(max(row - R, 0), min(row + R + 1, height))
        # X, Y = np.meshgrid(x, y)
        # R = np.sqrt(((X - column)**2 + (Y - row)**2))
        # mask = R < radius   
        # print("Mask: ")
        # print(mask)
        # while j < mask:
        #     if (mask[j] == True):
        #         index = index(j)
        #         accumulator[index] = 0  
        #     j = j + 1    
        
        theta = theta_range[theta_idx]
        values = (rho, theta)
        lines.append(values)
        accumulator[rho_idx, theta_idx] = 0
        i = i + 1    

    return lines