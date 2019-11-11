# Computer-Vision

This repository includes any projects that I have done online classes (Rajeev Ratan). Those projects primarily cover over Mathematics, traditional Machine Learning algorithms (Support Vector Machine), Deep Learning such as CNN (Covolutionary Neural Network), 

#### Basics of Computer Vision and OpenCV

    1/ Grayscalling:    finding gradient on edge points of an image
    2/ Colors HSV-Hue:  extracting a colored object
    3/ Color Gray-RGB:
    4/ Histogram:       helping us to understand the distribution behind the color of an image
    5/ Drawings of Shapes: being useful for object detection

#### Image Manipulation and Processing
    
    1/ Transformation:      including rotation, scaling, bitwise, pixel manipulation, edge detection
    2/ Image Translation:   moving the image over designated area in order to receive a part of image.
    3/ Rotation:            using this technique as data augmentation in case we run out of sources.
    4/ Scaling:             zooming would make image blurry, scalling help us overcome this drawback.
    5/ Image Pyramids (Resize):     resizing images as our expectation
    6/ Region of interest (Crop):   being useful when applying to Self-Driving-Cars as we teach car to concentrate on particular land part.
    7/ Bitwise (Mask):      being similar to logic gates with 1 and 0 
    8/ Sharpening:          enhancing the edge, corner of an image if it seems blur
    9/ Thresholding:        historical documents, where there is huge intensity difference between text and background, need transforming to be read by human. 
    10/ Dilation & Erosion: Dialtion adds pixels to image's edge, whereas Rosion removes pixels around image.
    11/ Edge Detection:     comparing point's pixel intensity to its neighborhood. If there is huge difference over 1 dimension, that is edge dectection. If there is huge difference over 2 dimensions, that is corner dectection.
    12/ Perspective & Affine Transforms: pinpointing the object corners to extract image.
     
#### Image Segmentation and Contours
    
    1/ Segmentation and Contours:   understanding of an image at the level of pixel.
    2/ Sorting Contours:            adjusting size/area.
    3/ Approx Contours & Convex:    convex hull (convex envelope/closure) includes a set of points in multiple dimensions.
    4/ Matching Contour:            finding locations, sizes, orientation of predefined objects in  an image. It can be done by identifying the shape or boundary of the objects.
    5/ Identify Contour:            being useful for shape analysis, object detection, object recogition.
    6/ Line Detection:              being used to detect blobs, based on different appearances, shapes, size, characteristics.
    7/ Counting Shapes:             finding Contours in the image, using approPolyDP function.
    
#### Object Detection 1

    1/ Feature Description: extracting useful information
    2/ Finding Waldo:       using the template to extract the desirable image's part by matching its pixel level. Being useful only when the template and the desirable image are identical.  
    3/ Finding Corners:     detecting the important features (known as interest points) in the image such as edges, corners, colors. 
    4/ HOGs:                being a representation of an image according to its intensity, gradient level(x, y direction) by applying kernels
    
#### Object Detection 2
    
    1/ HAAR Cascade:            Machine Learning learns positive images against negative ones, and detect object in other images
    2/ Face and Eye Detection:  using HAAR Cascade pretrained model to detect objects: face, eye
    3/ Car Video Detection:     using HAAR Cascade pretrained model to detect objects: car
    4/ Pedestrian Video Detection: using HAAR Cascade pretrained model to detect objects: pedestrian

#### Machine Learning and Deep Learning Using OpenCV

    1/ Handwritten Digit Recognition:   training deep learning model to classify handwritten digit number
    2/ Credit Card Reader:
    3/ Facial Recognition:      
    4/ Object recognition:      using pre-trained model of Yolo3, derived from DarkNet and DarkFlow, to detect objects: car, bus, people
    
#### Additional Knowledge taken away from the course
    
    1/ Visualing what CNN sees & Filter visualization
        Activation Maximization
        Saliency Map
        Heat Map
    2/ Image Classification
        Lenet, AlexNet, Google LeNet, 
        Inception, VGG(16, 19), ResNet (50, 101, 152)
    3/ Transfer Learning - Fine Tuning - Continuous Learning
    4/ Image Segmentation
        U Net
        IoU (Intersection over Union)
    5/ Object Detection
        Sliding window with HOG
        RCNN, Fast RCNN, Faster RCNN, Mask RCNN
        Single Shot Detector
        Yolo
    6/ Generative Adversarial Network (GAN)
