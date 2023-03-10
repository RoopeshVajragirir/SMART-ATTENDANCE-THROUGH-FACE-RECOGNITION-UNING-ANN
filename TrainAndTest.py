import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Dropout, Flatten
from keras.optimizers import Adam
########################################

path='images'
images=[]
classNo=[]
testRatio=0.2
valRatio=0.2
imageDimension=(32,32,3)

#########################################

myList=os.listdir(path)

classLength=len(myList)


print(classLength)

print("Importing Classes..........")
for x in range(0, classLength):
	myPicList=os.listdir(path+"/"+str(x))
	# myData/0/img.jpg
	for mypics in myPicList:
		curImg=cv2.imread(path+"/"+str(x)+"/"+mypics)
		curImg=cv2.resize(curImg,(imageDimension[0],imageDimension[1]))
		images.append(curImg)
		classNo.append(x)
	print(x)

images=np.array(images)
classNo=np.array(classNo)

xAxisTrain, xAxisTest, yAxisTrain, yAxisTest=train_test_split(images, classNo, test_size=testRatio)
xTrain, xAxisValidation, yTrain, yAxisValidation=train_test_split(xAxisTrain, yAxisTrain, test_size=valRatio)


print(xTrain.shape)

sample=[]

for x in range(0,classLength):
	sample.append(len(np.where(yTrain==x)[0]))


plt.figure(figsize=(10,5))
plt.bar(range(0, classLength),sample)
plt.title("Bar Plot of Classes & Images")
plt.xlabel("No Of Classes")
plt.ylabel("No of Images")
plt.show()


def preprocessing(image):
	# img=np.astype("uint8")
	image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image=cv2.equalizeHist(image)
	image=image/255
	return image


xTrain=np.array(list(map(preprocessing, xTrain)))
xTest=np.array(list(map(preprocessing, xAxisTest)))
xValidation=np.array(list(map(preprocessing, xAxisValidation)))


x_train=xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2],1)
x_test=xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2],1)
x_validation=xValidation.reshape(xValidation.shape[0], xValidation.shape[1], xValidation.shape[2],1)


imageDataGenerator=ImageDataGenerator(
	width_shift_range=0.1,
	height_shift_range=0.1,
	zoom_range=0.2,
	shear_range=0.1,
	rotation_range=10)

imageDataGenerator.fit(x_train)

y_train=to_categorical(yTrain, classLength)
y_test=to_categorical(yAxisTest, classLength)
y_validation=to_categorical(yAxisValidation, classLength)


def myModel():
	sizeOfFilter1=(3,3)
	sizeOfFilter2=(3,3)
	sizeOfPool=(2,2)

	model=Sequential()
	model.add((Conv2D(32, sizeOfFilter1, input_shape=(imageDimension[0],imageDimension[1],1),activation='relu')))
	model.add((Conv2D(32, sizeOfFilter1,activation='relu')))
	model.add(MaxPooling2D(pool_size=sizeOfPool))

	model.add((Conv2D(64, sizeOfFilter2,activation='relu')))
	model.add((Conv2D(64, sizeOfFilter2,activation='relu')))
	model.add(MaxPooling2D(pool_size=sizeOfPool))
	model.add(Dropout(0.5))


	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(classLength, activation='softmax'))
	model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
	return model

model=myModel()
print(model.summary())

result=model.fit_generator(imageDataGenerator.flow(x_train, y_train,batch_size=50),
	steps_per_epoch=1000,
	epochs=2,
	validation_data=(x_validation,y_validation),
	shuffle=1)

model.save("MyTrainingModel.h5")