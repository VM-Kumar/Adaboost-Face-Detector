# Ada-boost Face Detector
**Goal:** To classify a given image as face image or non-faceimage. This method uses multiple weak features to obtain a strong feature to detect facial presence.

![github_image1](https://user-images.githubusercontent.com/70597312/103395445-385eb900-4b54-11eb-8fb6-bff5290e43bd.PNG)

The content is organized as follows:
1. [description_report.pdf](https://github.com/VM-Kumar/Adaboost-Face-Detector/blob/main/description_report.pdf) : detailed descriptio of the implementation of adaboost algorithm and results obtained on the chosen dataset.
2. [data_acquisition.py](https://github.com/VM-Kumar/Adaboost-Face-Detector/blob/main/data_acquisition.py) : This code is used to obtain face images from raw face data and its annotations from FDDB dataset. To use this code download the face data and annotations from FDDB data set from this link [dataset](http://viswww.cs.umass.edu/fddb/index.html#download) . Use the path to the annotations file and images folder in the data_acquisition code as indicated in the code comments. Also give the paths for destination folders for face and non-face images as indicated in the code comments. Now run the code to obtain required face and non-face dataset.
3. [https://github.com/VM-Kumar/Adaboost-Face-Detector/blob/main/Adaboost_detection_Final_Code.py](https://github.com/VM-Kumar/Adaboost-Face-Detector/blob/main/Adaboost_detection_Final_Code.py) : final code to implement Adaboost algorithm and to test it to obtain detection results. Give the path to the folders obtained from data_acquisition as indicated in code comment. These folders contain the required training and test face and non_face images.
**note:** face data is obtained from image's annotation whereas non-face images are obtained by randomly cropping from the images background.


