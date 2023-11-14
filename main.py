import os
import cv2
import pickle
import face_recognition
import numpy as np
capture = cv2.VideoCapture(0)
capture.set(3,1280)
capture.set(4,720)

# Importing the modes
folderModePath = 'resources/Modes'
imgPathList = os.listdir(folderModePath)
imgModeList = []

for path in imgPathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

backgroundImage = cv2.imread('resources/background.png')

# Load the encoding file
print("Loading encoding file")
file = open('Encode.p','rb')
encodingListWithIds = pickle.load(file)
file.close
encodeListKnown, studentIds = encodingListWithIds
print("Loading Encode completed")

while True:
    success, img = capture.read()

    imgSmall =cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_RGB2BGR)

    faceCurrentFrame = face_recognition.face_locations(imgSmall)
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall,faceCurrentFrame)


    backgroundImage[162:162+480,55:55+640] = img
    backgroundImage[44:44+633,808:808+414] = imgModeList[0]

    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("Matches: ",matches)
        print("Face Distance: ",faceDistance)
        matchIndex = np.argmin(faceDistance)
        print("Match Index", matchIndex)

    # cv2.imshow("WebCam",img)
    cv2.imshow("Student Attendance", backgroundImage)
    cv2.waitKey(1)