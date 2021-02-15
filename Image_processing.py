"""

Created by Arijus Grotuzas.
This is an Image_processing.py file that contains the image_analysis class and its methods.
For actual usage of the class and it's methods please refer to the Main.py file.
NOTE: This class uses Numba and is actually a jitclass, for more information about Numba please refer to https://numba.pydata.org/numba-doc/dev/user/
If any issues arise with this code refer to the README file.

"""

from numba import int32
from numba import jitclass
import numpy as np
import math

spec = [('value', int32)]


@jitclass(spec)
class image_analysis(object):
    def __init__(self):
        self.value = 0

    def intensity(self, image):
        (height, width, channels) = image.shape
        Intensity = np.zeros((height, width), np.uint8)  # creating an array for a new image
        for i in range(height):  # two for loops that go through all the pixels of the image
            for j in range(width):
                (r, g, b) = image[i, j]  # retrieving a pixel value at a certain point in the image

                x = int(r)
                y = int(g)
                z = int(b)
                RGB = x + y + z  # adding up all the rgb values

                inten = int(RGB / 3)  # splitting the RGB sum by 3

                Intensity[i, j] = inten  # assigning new pixel values to the corresponding place in the array
        return Intensity  # returns the array of the new image

    def threshold(self, image, T1, T2):
        (height, width) = image.shape
        thresholded = np.zeros((height, width), np.uint8)

        for i in range(height):  # two for loops that go through all the pixels of the image
            for j in range(width):
                p = image[i, j]

                if T1 < p < T2:
                    p = 255
                else:
                    p = 0

                thresholded[i, j] = p
        return thresholded

    def treshold_color(self, image, argument):
        (height, width, channels) = image.shape
        thresholded = np.zeros((height, width, 3), np.uint8)

        for i in range(height):  # two for loops that go through all the pixels of the image
            for j in range(width):
                (r, g, b) = image[i, j]

                if argument == 'b':
                    diff = (g + b) / (r + 1)

                    if diff > 0.8:
                        r = 0
                        b = 0
                        g = 0
                    else:
                        g = 0
                        b = 0
                        r = 255

                elif argument == 'g':
                    diff = (r + b) / (g + 1)

                    if diff > 0.95:
                        r = 0
                        b = 0
                        g = 0
                    else:
                        r = 0
                        b = 0
                        g = 255

                elif argument == 'r':
                    diff = (g + r) / (b + 1)

                    if diff > 0.8:
                        r = 0
                        b = 0
                        g = 0
                    else:
                        r = 0
                        g = 0
                        b = 255

                thresholded[i, j] = (r, g, b)
        return thresholded

    def histogram_stretching(self, image):
        (height, width) = image.shape
        stretched = np.zeros((height, width), np.uint8)
        minimum = np.amin(image)
        maximum = np.amax(image)

        for i in range(height):
            for j in range(width):
                p = image[i, j]
                g = ((p - minimum) / (maximum - minimum)) * 255
                stretched[i, j] = g
        return stretched

    def histogram_calculation(self, image):
        (height, width) = image.shape
        hist_values = np.zeros(256, np.uint8)

        for i in range(height):
            for j in range(width):
                p = image[i, j]
                hist_values[p] = hist_values[p] + p
        return hist_values

    def contrast(self, image, a):
        (height, width) = image.shape
        contrasted = np.zeros((height, width), np.uint8)

        for i in range(height):
            for j in range(width):
                p = image[i, j]
                g = a * p

                if g > 255:
                    g = 255

                contrasted[i, j] = g
        return contrasted

    def brightness(self, image, a):
        (height, width) = image.shape
        brightened = np.zeros((height, width), np.uint8)
        for i in range(height):
            for j in range(width):
                p = image[i, j]

                p = p + a

                if p < 0:
                    p = 0

                elif p > 255:
                    p = 255

                brightened[i, j] = p
        return brightened

    def normalizedRGB(self, image):
        (height, width, channels) = image.shape
        normalized = np.zeros((height, width, 3), np.uint8)
        for i in range(height):  # two for loops that go through all the pixels of the image
            for j in range(width):
                (r, g, b) = image[i, j]

                x = int(r)
                y = int(g)
                z = int(b)

                s = x + y + z

                if s == 0:
                    s = 1

                x = x / s * 255
                y = y / s * 255
                z = z / s * 255

                normalized[i, j] = (x, y, z)
        return normalized

    def median_filter_grayscale(self, image):
        (height, width) = image.shape
        filtered = np.zeros((height, width), np.uint8)
        a = np.zeros(9, np.uint8)
        for i in range(1, height - 1):  # two for loops that go through all the pixels of the image
            for j in range(1, width - 1):
                a[0] = image[i - 1, j - 1]
                a[1] = image[i, j - 1]
                a[2] = image[i + 1, j - 1]
                a[3] = image[i - 1, j]
                a[4] = image[i, j]
                a[5] = image[i + 1, j]
                a[6] = image[i - 1, j + 1]
                a[7] = image[i, j + 1]
                a[8] = image[i + 1, j + 1]

                p = np.median(a)

                filtered[i, j] = p
        return filtered

    def median_blur_grayscale(self, image, size):
        (height, width) = image.shape
        filtered = np.zeros((height, width), np.uint8)
        iterate = self.ceiling_division(size, 3)

        for idx, x in np.ndenumerate(image):
            filtered[idx[0], idx[1]] = self.median_2d(
                image[idx[0] - iterate:idx[0] + iterate, idx[1] - iterate: idx[1] + iterate])
        return filtered[size:height - size, size:width - size]

    def draw_line(self, image, thickness, StartX, StartY, EndX, EndY):

        Xdist = EndX - StartX
        Ydist = EndY - StartY

        m = Ydist / Xdist

        for x in np.nditer(range(StartX, EndX)):
            y = m * x + StartY
            y = int(round(y))

            for t in range(thickness):
                image[y + t, x] = (0, 255, 0)
                image[y - t, x] = (0, 255, 0)

        return image

    def median_2d(self, arr):
        if arr.size != 0:
            m = arr.flatten()
            return np.median(m)
        else:
            return 0

    def ceiling_division(self, a, b):
        return -(-a // b)

    def correlate(self, image, kernel, norm):
        (height, width) = image.shape
        correlated = np.zeros((height, width), np.uint8)
        array = image
        for i in range(1, height - 1):  # two for loops that go through all the pixels of the image
            for j in range(1, width - 1):
                f1 = array[i - 1, j - 1]
                f2 = array[i, j - 1]
                f3 = array[i + 1, j - 1]
                f4 = array[i - 1, j]
                f5 = array[i, j]
                f6 = array[i + 1, j]
                f7 = array[i - 1, j + 1]
                f8 = array[i, j + 1]
                f9 = array[i + 1, j + 1]

                x = 1
                y = 1

                h1 = kernel[x - 1, y - 1]
                h2 = kernel[x, y - 1]
                h3 = kernel[x + 1, y - 1]
                h4 = kernel[x - 1, y]
                h5 = kernel[x, y]
                h6 = kernel[x + 1, y]
                h7 = kernel[x - 1, y + 1]
                h8 = kernel[x, y + 1]
                h9 = kernel[x + 1, y + 1]

                result = (f1 * h1) + (f2 * h2) + (f3 * h3) + (f4 * h4) + (f5 * h5) + (f6 * h6) + (f7 * h7) + (
                        f8 * h8) + (
                                 f9 * h9)
                result = result / norm

                correlated[i, j] = result

        return correlated

    def correlate_new(self, image, kernel, R, norm):
        (height, width) = image.shape
        correlated = np.zeros((height, width), np.uint8)
        # result = 0
        for i in range(1, height - 1):  # two for loops that go through all the pixels of the image
            for j in range(1, width - 1):
                result = 0
                for k in range(-R, R + 1):
                    for l in range(-R, R + 1):
                        result = result + kernel[l, k] * image[i + k, j + l]

                correlated[i, j] = result / norm
        return correlated

    def alpha_blending(self, image1, image2, alpha):
        (height1, width1, channels1) = image1.shape

        crop_image2 = image2[0:height1, 0:width1]
        blended = np.zeros((height1, width1, 3), np.uint8)
        for i in range(height1):
            for j in range(width1):
                (x1, y1, z1) = image1[i, j]
                (x2, y2, z2) = crop_image2[i, j]

                r = alpha * x1 + (1 - alpha) * x2
                g = alpha * y1 + (1 - alpha) * y2
                b = alpha * z1 + (1 - alpha) * z2

                blended[i, j] = (r, g, b)

        return blended

    def mirror(self, image):
        (height, width, channels) = image.shape
        mirrored = np.zeros((height, width, 3), np.uint8)
        for i in range(height):
            for j in range(width):
                (x, y, z) = image[i, j]

                mirrored[i, width - j] = (x, y, z)

        return mirrored

    def half_mirrored(self, image):
        (height, width, channels) = image.shape
        half_mirror = np.zeros((height, width, 3), np.uint8)

        for i in range(height):  # two for loops that go through all the pixels of the image
            for j in range(width // 2 + 1):
                (x, y, z) = image[i, j]

                half_mirror[i, j] = (x, y, z)
                half_mirror[i, width - j] = (x, y, z)

        return half_mirror

    def inverse(self, image):
        (height, width) = image.shape
        inversity = np.zeros((height, width), np.uint8)  # creating an array for a new image
        for i in range(height):  # two for loops that go through all the pixels of the image
            for j in range(width):
                # retrieving a pixel value at a certain point in the image
                pixel = 255 - image[i, j]

                inversity[i, j] = pixel  # assigning new pixel values to the corresponding place in the array
        return inversity  # returns the array of the new image

    def saturation(self, image):
        (height, width, channels) = image.shape
        Saturation = np.zeros((height, width), np.uint8)  # creating an array for a new image
        for i in range(height):  # two for loops that go through all the pixels of the image
            for j in range(width):
                (r, g, b) = image[i, j]  # retrieving a pixel value at a certain point in the image

                x = int(r)
                y = int(g)
                z = int(b)

                min1 = min(x, y, z)
                max1 = max(x, y, z)
                d = max1 - min1

                if max1 != 0:
                    S = d / max1  # getting the saturation value
                else:
                    S = 0  # in case where value is 0

                S = S * 255  # scaling it up to the pixel values

                Saturation[i, j] = S  # assigning new pixel values to the corresponding place in the array
        return Saturation  # returns the array of the new image

    def hue_old(self, image):
        (height, width, channels) = image.shape
        Hue = image.copy()  # copying an image to get an array of rgb values
        for i in range(height):  # two for loops that go through all the pixels of the image
            for j in range(width):
                (b, g, r) = image[i, j]  # retrieving a pixel value at a certain point in the image

                x = int(r)
                y = int(g)
                z = int(b)

                minimum = min(x, y, z)
                maximum = max(x, y, z)
                d = maximum - minimum

                if d == 0:
                    d = 1

                if x == maximum:
                    H = (y - z) / d  # getting the hue value when R is the highest value
                elif y == maximum:
                    H = 2 + (z - x) / d  # getting the hue value when G is the highest value
                else:
                    H = 4 + (x - y) / d  # getting the hue value when B is the highest value

                H *= 60  # turning the hue value to degrees
                if H < 0:
                    H *= 360

                H = H / 360  # turning hue value into a ratio
                rRatio = x / 255
                gRatio = y / 255
                bRatio = z / 255

                if x == maximum:
                    Hue[i, j] = [(rRatio + H) * 255, y,
                                 z]  # changing the value of R in the corresponding place in the array
                elif y == maximum:
                    Hue[i, j] = [x, (gRatio + H) * 255,
                                 z]  # changing the value of R in the corresponding place in the array
                else:
                    Hue[i, j] = [x, y,
                                 (bRatio + H) * 255]  # changing the value of R in the corresponding place in the array

        return Hue  # returns the array of the new image

    def hue_new(self, image):
        (height, width, channels) = image.shape
        hu = np.zeros((height, width), np.uint8)

        for i in range(height):
            for j in range(width):
                (b, g, r) = image[i, j]  # retrieving a pixel value at a certain point in the image
                h = 0

                nomi = (0.5 * (r - g) + (r - b))
                denom = pow((pow((r - g), 2) + ((r - b) * (g - b))), 0.5)

                theta = np.arccos(nomi / denom)

                if b <= g:
                    h = theta
                elif b > g:
                    h = 360 - theta

                hu[i, j] = h

        return hu

    def standard_deviation(self, arr):
        mean = int(round(float(np.mean(arr))))
        diff = []
        var = 0

        for i in arr:
            diff.append((mean - i))

        for i in diff:
            var = var + pow(i, 2)

        var = int(round(var / len(diff)))
        stand = int(round(math.sqrt(var)))

        return stand
