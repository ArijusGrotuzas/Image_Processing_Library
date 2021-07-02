# Image_Processing_Library
This is a library that feature many different image processing algorithms for analysing and dissecting long image sequences. Project features two files and one class.
The Main.py file contains example of using the image_analysis class, while the Image_processing.py file contains the only the image analysis class.
The image_analysis class methods for image analysis. Some of these methods are so far only applicable only either to RGB or Grayscale images. 
In addition, the precision of images that the methods support is 8 bpp. RAW image files are not supported by any of the methods, yet.
This code is a work in progress and should be treated as such.

The image_analysis class features algorithms for:
- RGB to Grayscale conversion (RGB input)
- Thresholding of the Grayscale images (Grayscale Input)
- Histogram stretching (Grayscale Input)
- Histogram calculation (Grayscale Input)
- Contrast adjustment (Grayscale Input)
- Brightness adjustement (Grayscale Input)
- Normalized RGB (RGB input)
- Median filter (Grayscale Input)
- Drawing lines (RGB input)
- Correlation with user specified kernel (Grayscale Input)
- Alpha blending (RGB input)
- Mirroring of the image (RGB input)
- Inversing image (Grayscale Input)
- Getting the saturation values of the image (RGB input)
- Getting the hue values of the image (RGB input)
- Calculating standard deviation (Array)

The repository features a small test video, if you wish to use any other different video, simply add it to the directory folder of the project and channge the code so that OpenCV loads your file.
