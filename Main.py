"""

Created by Arijus Grotuzas.
This is a Main.py file that simply showcases how the image_analysis class can be used.
Some of the methods are only limited to grayscale images and some methods are only limited to the rgb images.
Please refer to the README file for which methods support grayscale and which support rgb.
This code just as the code in the Image_processing file are work in progress and should be treated as such.
If any issues arise with this code or the image_analysis class please refer to the README file.

"""

from Image_processing import image_analysis
import numpy as np
import cv2


def main():
    gaussian_edge = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]])

    gaussian_blur = np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])

    mean_blur = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])

    sobel_edge_vertical = np.array([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]])

    sobel_edge_horizontal = np.array([[1, 0, -1],
                                      [2, 0, -2],
                                      [1, 0, -1]])

    cap = cv2.VideoCapture('test.mp4')
    processor = image_analysis() # Initializing the object for image processing

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Video ended, exiting program...")
            break

        """ Example showcasing the usage of the image_analysis class and it's methods"""
        gray = processor.intensity(frame)  # Turning images to grayscale

        lowered_brightness = processor.brightness(gray, -90)  # Turning down the brightness of the image

        stretched_histogram = processor.histogram_stretching(lowered_brightness)  # Stretching the histogram of the image

        thresh = processor.threshold(stretched_histogram, 20, 100)  # Threshold the image

        edge_detect_vertical = processor.correlate(thresh, sobel_edge_vertical, 1)  # Vertical edge detection
        edge_detect_horizontal = processor.correlate(thresh, sobel_edge_horizontal, 1)  # Horizontal edge detection

        final_edge_detect = edge_detect_horizontal + edge_detect_vertical  # Final edge detection

        """Using OpenCV's method for showing the image sequence"""
        cv2.imshow('Adjusted image', final_edge_detect)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
