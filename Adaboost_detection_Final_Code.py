import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import math
import cv2
from PIL import Image
from math import *
from dask import delayed
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature


########################### Extracting and storing face,nonface,training and testing images#########################
img_face_train=[]
for i in range(1000):
    pat=r'C:\Users\venkatesh\Desktop\data\dataset_face_train\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img_face_train.append(image)
img_face_train=np.array(img_face_train)    


img_face_test=[]
for i in range(100):
    pat=r'C:\Users\venkatesh\Desktop\data\dataset_face_test\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img_face_test.append(image)
img_face_test=np.array(img_face_test)


img_nonface_train=[]
for i in range(1000):
    pat=r'C:\Users\venkatesh\Desktop\data\dataset_nonface_train\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img_nonface_train.append(image)
img_nonface_train=np.array(img_nonface_train)


img_nonface_test=[]
for i in range(100):
    pat=r'C:\Users\venkatesh\Desktop\data\dataset_nonface_test\im{}.jpg'.format(i+1)
    image=cv2.imread(pat)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    img_nonface_test.append(image)
img_nonface_test=np.array(img_nonface_test)
print('Data Acquisition....')


###################################### HAAR Feature Extraction #################################################### 
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],feature_type=feature_type,feature_coord=feature_coord)


images=np.append(img_face_train,img_nonface_train,axis=0)
feature_types = ['type-2-x', 'type-2-y','type-3-x','type-3-y']
X = delayed(extract_feature_image(img, feature_types) for img in images)
X = np.array(X.compute(scheduler='threads'))
print('Feature Extraction....')

############################ Visualizing Top 10 features before boosting ############################################
#initialization before boosting
no_images=X.shape[0]
no_features=X.shape[1]
iterations=100
y = np.array([1] * img_face_train.shape[0] + [-1] *img_nonface_train.shape[0])
W=(1/no_images)*np.ones(no_images)
alpha=[]
h_index=[]
E=np.zeros(no_features)

#obtaining h from features
decision_threshold=np.mean(X,axis=0)
temp= X>decision_threshold
h=2*temp-1

#coordinates of all possible features 
feature_coord, feature_type = haar_like_feature_coord(width=images.shape[2], height=images.shape[1],feature_type=feature_types)
before_test=[]
for i in range(no_images):
    x=h[i]!=y[i]
    before_test.append(x)
print(len(before_test))
print(len(before_test[0]))
before_test=np.sum(before_test,axis=0)
a=np.array(before_test)
print(a.shape)
a.argsort()[:10]
fig1, axes = plt.subplots(5, 2)
for idx, ax in enumerate(axes.ravel()):
    image = images[2]
    image = draw_haar_like_feature(image, 0, 0,images.shape[2],images.shape[1],[feature_coord[a[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
_ = fig1.suptitle('The most important features(before boosting)')
fig1.show()



################################ Training and Ada Boost Implemementation ##########################################
#getting I from h
I=np.zeros(h.shape)
for j in range(no_features):
        I[:,j]=h[:,j] != y
        
#adaboost algorithm implementation
for i in range(iterations):
    for j in range(no_features):
        E[j]=(np.dot(I[:,j],W))
    x = [k for k in range(no_features) if k not in h_index]
    min=100
    for count in x:
        if E[count]<min:
            min=E[count]
            index=count
    h_index.append(index)
    error=E[index]
    a= 0.5*math.log((1-error)/error)
    alpha.append(a)
    W=np.multiply(W,np.exp(-1*a*np.multiply(h[:,index],y)))
    W=(1/W.sum())*W
    print(i)

    
####################### Visualizing first 10 features selected after boosting ####################################
#plotting only then first 10 features after boosting    
fig2, axes = plt.subplots(5, 2)
for idx, ax in enumerate(axes.ravel()):
    image = images[2]
    image = draw_haar_like_feature(image, 0, 0,images.shape[2],images.shape[1],[feature_coord[h_index[idx]]])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
_ = fig2.suptitle('First 10 features after boosting')
fig2.show()



#################################### Testing and ROC plot ########################################################
#obtaining and feature extraction of test images
images_test=np.append(img_face_test,img_nonface_test,axis=0)
T = delayed(extract_feature_image(img, feature_types) for img in images_test)
T = np.array(T.compute(scheduler='threads'))
no_images_test=T.shape[0]
print('Feature extraction for testing....')

#getting h of test images 
decision_threshold1=np.mean(T,axis=0)
temp1= T>decision_threshold
h1=2*temp1-1

#before testing
y_test=[1]*(int(no_images_test/2)) + [-1]*(int(no_images_test/2))
H=[]

#obtaining testing result H
for i in range(no_images_test):
        a=0
        for j in range(iterations):
                a=a+alpha[j]*h1[i][h_index[j]]
        H.append(a)
        
#test accuracy calculation
term1 =np.array(H)
temp2= term1>0
temp2=2*temp2-1
temp2=np.equal(temp2,y_test)
print('number of True Positives=',np.sum(temp2[:int(no_images_test/2)]))
print('number of True Negatives=',np.sum(temp2[int(no_images_test/2):]))
print('number of False Positives=',int(no_images_test/2)- np.sum(temp2[:int(no_images_test/2)]))
print('number of False Negatives=',int(no_images_test/2)- np.sum(temp2[int(no_images_test/2):]))
print('accuracy',(np.sum(temp2)/no_images_test)*100)

#Comparison for different thresholds
no_roc = 100
threshold = np.linspace(np.min(term1), np.max(term1), no_roc)
TP = []
TN = []
FP = []
FN = []
for k in range(len(threshold)):
	TP.append(term1[:int(no_images_test/2)] >= threshold[k])
	FN.append(term1[:int(no_images_test/2)] < threshold[k])
	TN.append(term1[int(no_images_test/2):no_images_test] < threshold[k])
	FP.append(term1[int(no_images_test/2):no_images_test] >= threshold[k])
TP = np.sum(TP, axis = 1)
TN = np.sum(TN, axis = 1)
FP = np.sum(FP, axis = 1)
FN = np.sum(FN, axis = 1)

#ROC plot
fig3=plt.figure(3)
plt.plot(FP/100, TP/100, marker='o')
plt.title('Receiver operating characteristic (ROC) curve')
plt.xlabel('False Positive Rate (1 - Specificity)')	
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlim(0,1)
plt.ylim(0,1)	
fig3.show()







