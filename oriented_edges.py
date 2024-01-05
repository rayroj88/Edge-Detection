import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from gradient_orientations import gradient_orientations

def oriented_edges(img, sigma, threshold, direction, tolerance):
    """
    Detects oriented edges in a grayscale or color image.

    Parameters:
    img (numpy.ndarray): The input image. If the image is in color, it will be converted to grayscale.
    sigma (float): The standard deviation of the Gaussian filter used to blur the image.
    threshold (float): The lower threshold for the Canny edge detector. The upper threshold is twice this value.
    direction (float): The desired direction of the edges, in degrees. The function will detect edges whose direction
        is within `tolerance` degrees from this direction.
    tolerance (float): The tolerance in degrees for the edge direction. The function will detect edges whose direction
        is within this tolerance from the `direction` parameter.

    Returns:
    numpy.ndarray: A binary image where the edge pixels are set to 255 and the non-edge pixels are set to 0.
    """
    
    #Function to rotate kernel direction degrees
    def rotate_kernel(kernel, direction):
        rows, cols = kernel.shape[0], kernel.shape[1]
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), direction, 1)
        return cv.warpAffine(kernel, M, (cols, rows))
    
    def blur_image(image, sigma_x, sigma_y):
        ksize = (int(2 * round(3 * sigma_x) + 1), int(2 * round(3 * sigma_y) + 1))
        return cv.GaussianBlur(image, ksize, sigma_x, sigmaY=sigma_y)
    
    #Generate kernel with Sobel Operator for x and y axis
    dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    dx = dx / np.sum(np.abs(dx))
    dy = dx.T
    
    #Rotate the kernel
    rot_x = rotate_kernel(dx, direction)
    rot_y = rotate_kernel(dy, direction)
    
    blurred_img = blur_image(img, sigma, sigma)
    
    #Canny edge detector
    img = cv.Canny(blurred_img, threshold, 2*threshold)
    
    #Handle directions larger than 180
    direction = direction % 180
    
    #Calculate gradient norms
    orientation = gradient_orientations(img)
    
    #separate positive angles and negative ones
    positive_gradient = orientation >= 0
    negative_gradient = orientation < 0
    
    #Create a matrix of size orientation of all zeroes
    edge_direction = np.zeros_like(orientation)
    
    #If gradient angle is positive, edge direction will be perpendicular to it so add 90 and % 180 to get edge direction
    #If gradient angle is negative, edge direction will be perpendicular to it so add 270 and % 180 to get edge direction
    edge_direction[positive_gradient] = (orientation[positive_gradient] + 90) % 180
    edge_direction[negative_gradient] = (orientation[negative_gradient] + 270) % 180
    
    #Keep edges that are within the tolerance
    #Did not account for opposite direction!! | np.abs(edge_direction- 180 - direction) <= tolerance
    edge_img = np.abs(edge_direction - direction) <= tolerance

    return edge_img
