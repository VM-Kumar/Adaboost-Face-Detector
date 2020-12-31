import cv2
from PIL import Image 

count=1  
with open(r'C:\Users\venkatesh\Desktop\data\FDDB-folds','r') as file1:
    Lines = file1.readlines() 

for i in range(len(Lines)):
    if Lines[i][0]=='1' and Lines[i][1]=='\n' and count<=1100:
        Lines[i-1] = Lines[i-1].rstrip("\n")
        # path to unannotated images folder
        source='.................../data/{}.jpg'.format(str(Lines[i-1]))
        specs=Lines[i+1].split()
        major=int(float(specs[0]))
        minor=int(float(specs[1]))
        centrex=int(float(specs[3]))
        centrey=int(float(specs[4]))
        img = cv2.imread(source)
        print(source)
        # according to annotations, centrex corresponds to columns and centrey corresponds to row
        if 2*major>=90 and 2*minor>=90:
            x1=centrey-major
            x2=centrey+major
            y1=centrex-minor
            y2=centrex+minor
            rows = len(img)
            cols = len(list(zip(*img)))
            x1 = 0 if x1 < 0 else x1
            x2 = rows if x2 >rows else x2
            y1 = 0 if y1 < 0 else y1
            y2 = cols if y2 >cols else y2
            final_face=img[x1:x2,y1:y2]
            final_nonface=img[0:100,0:100]
            if count<=1000:
                #destination folder1: training face dataset
                str1='..................../dataset_face_train/im{}.jpg'.format(count)
                #destination folder2: training non-face dataset
                str2='................./dataset_nonface_train/im{}.jpg'.format(count)
            else:
                #destination folder3: test face dataset
                str1='................/dataset_face_test/im{}.jpg'.format(count-1000)
                #destination folder4: test non-face dataset
                str2='............./dataset_nonface_test/im{}.jpg'.format(count-1000)
##            final_face=cv2.cvtColor(final_face,cv2.COLOR_RGB2GRAY)
##            final_nonface=cv2.cvtColor(final_nonface,cv2.COLOR_RGB2GRAY)
            final_face=cv2.resize(final_face,(20,20),interpolation=cv2.INTER_AREA)
            final_nonface=cv2.resize(final_nonface,(20,20),interpolation=cv2.INTER_AREA)
    
            cv2.imwrite(str1,final_face)
            cv2.imwrite(str2,final_nonface)




            
            count=count+1
print(count)

