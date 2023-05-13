import cv2
import numpy as np

class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape

def color1():

# Define a sample BGR color value
    bgr_color = (30.282038594755075, 243.16922315685306, 185.6011875309253)

# Convert BGR color to RGB color
    rgb_color = cv2.cvtColor(np.array([bgr_color], dtype=np.uint8), cv2.COLOR_BGR2RGB)[0][0]

# Print RGB color values
    print("RGB color values:", rgb_color)

# Convert RGB color to HSV color
    hsv_color = cv2.cvtColor(np.array([[rgb_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]

# Print HSV color values
    print("HSV color values:", hsv_color)

# Load the two images
img1 = cv2.imread('images/img1.PNG')

gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('images/img2.PNG')

image = img1.copy()

# get the size of the largest image
max_height = max(img1.shape[0], img2.shape[0])
max_width = max(img1.shape[1], img2.shape[1])

# resize the images to the same size
img1 = cv2.resize(img1, (max_width, max_height))
img2 = cv2.resize(img2, (max_width, max_height))

"""
desired_width = 800
height ,width = image.shape[:2]
aspect_ratio = width/height
new_height = int(desired_width/aspect_ratio)
new_size = (desired_width,new_height)
img1 = cv2.resize(img1,new_size)

desired_width = 800
height ,width = image.shape[:2]
aspect_ratio = width/height
new_height = int(desired_width/aspect_ratio)
new_size = (desired_width,new_height)
img2 = cv2.resize(img2,new_size)
"""

# Find the absolute difference between the two images
diff = cv2.absdiff(img1, img2)

# Convert the difference image to grayscale and threshold it
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]

# Perform morphological operations to remove noise and smooth the object boundaries
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.erode(thresh, kernel, iterations=1)
thresh = cv2.dilate(thresh, kernel, iterations=1)

# Find the contours of the objects in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

missing_objcet = 0
# Compare the contours of the objects in the two images to identify the differences
for contour in contours:
    cv2.drawContours(img1, contours, -1, (0,0,0), 3)
    area = cv2.contourArea(contour)
    if area > 100:  # Ignore small contours
        missing_objcet +=1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        
        print("Area of Missing Object : ",area)

        perimeter = cv2.arcLength(contour,True)
        print("Perimeter of Missing Object : ",perimeter)

        sd = ShapeDetector()
        shape = sd.detect(contour)
        print("Shape of Missing Object : ",shape)

        x, y, w, h = cv2.boundingRect(contour)
        crop = image[y:y+h, x:x+w]
        color = cv2.mean(crop)[:3]
        print('Color of Missing object is : ', color)

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask,[contour],-1,255,-1)
        mean_color = cv2.mean(img1,mask=mask)[:3]
        print("Color of contour is : ",mean_color )
        
        aspect_ratio = float(w)/h
        print("Aspect ratio of Missing Object : ",aspect_ratio)
        
        rect_area = w*h
        extent = float(area)/rect_area
        print("Extent of Missing Object : ",extent)

        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        print("Solidity of Missing Object : ",solidity)

        moments = cv2.moments(contour)
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        bbox = cv2.boundingRect(contour)

        print("Centroid of Missing object ({},{})".format(centroid_x,centroid_y))
        print("Bbox of Missing Object : ",bbox)

        # Texture

        mask = np.zeros_like(img2)
        cv2.drawContours(mask,[contour],0,255,-1)
        blur = cv2.GaussianBlur(mask,(5,5),0)
        texture = cv2.bitwise_and(img1,blur)
        #cv2.imshow('Texture of Missing objects', texture)
        #cv2.waitKey(0)


print("Count Of Missing Object : ",missing_objcet)       
#color1()
# Display the results
cv2.imshow('Image 1', img1)
cv2.waitKey(0)
cv2.imshow('Image 2', img2)
cv2.waitKey(0)
cv2.imshow('Missing Object', thresh)
cv2.waitKey(0)


