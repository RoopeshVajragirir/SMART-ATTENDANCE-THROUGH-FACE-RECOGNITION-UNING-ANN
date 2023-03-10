# Full Project Code Mail : vatshayan007@gmail.com
# If you get error then Mail : vatshayan007@gmail.com

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas

path = 'Images_Attendance'
images = []
names = []
myList = os.listdir(path)
print(myList)
for list in myList:
    curvedImg = cv2.imread(f'{path}/{list}')
    images.append(curvedImg)
    names.append(os.path.splitext(list)[0])
print(names)

dataRead = pandas.read_csv ('Attendance.csv')
print(dataRead)

def closeWindow():
    capture.release()

def searchEncodings(images):
    encode =[]
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faceEncodings = face_recognition.face_encodings(image)[0]
        encode.append(faceEncodings)
    return encode

def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        readData = f.readlines()
        names = []
        for rows in readData:
            values = rows.split(',')
            names.append(values[0])
        if name not in names:
            timeNow = datetime.now()
            time = timeNow.strftime('%H:%M:%S')
            date = timeNow.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{time},{date}')

encodeList = searchEncodings(images)
print('Encoding Complete')

capture = cv2.VideoCapture(0)

while True:
    success, image1 = capture.read()
    imageSize= cv2.resize(image1, (0, 0), None, 0.25, 0.25)
    imageColor = cv2.cvtColor(imageSize, cv2.COLOR_BGR2RGB)

    facesCurvedFrame = face_recognition.face_locations(imageSize)
    encodesCurvedFrame = face_recognition.face_encodings(imageSize, facesCurvedFrame)

    condition = False
    Id=0
    for encodeFace, faceLoc in zip(encodesCurvedFrame, facesCurvedFrame):
        compareFaces = face_recognition.compare_faces(encodeList, encodeFace)
        faceDistance = face_recognition.face_distance(encodeList, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)
        Id = Id+1
        if compareFaces[matchIndex]:
            name = names[matchIndex].upper()
            print(name," is present")
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image1, (x1, y2-35), (x2, y2), (0, 250, 0), cv2.FILLED)
            cv2.putText(image1, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)
    cv2.imshow('webcam', image1)
    if cv2.waitKey(10) == 13:
        break
capture.release()
cv2.destroyAllWindow()




