import cv2
import face_recognition
import pickle
import os

# Importing the student images
studentImagesPath = 'images'
imgPathList = os.listdir(studentImagesPath)
imgList = []
studentIds = []
for imgPath in imgPathList:
    imgList.append(cv2.imread(os.path.join(studentImagesPath,imgPath)))
    studentId = os.path.splitext(imgPath)[0]
    studentIds.append(studentId)

# generate encodings
def findEndodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encode Started")
encodeListKnown = findEndodings(imgList)
encodingListWithIds = [encodeListKnown, studentIds] 
print("Encode Completed")
file = open('Encode.p','wb')
pickle.dump(encodingListWithIds,file)
file.close()
print("File Saved")