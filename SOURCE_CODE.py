import os
import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet
import keras
from keras import ops
from tensorflow.keras.utils import custom_object_scope # type: ignore

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2 # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Set the environment variable
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reconfigure the stdout to use UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')


df = pd.read_csv(r'C:\Users\dhruv\Downloads\NUMBER PLATE DETECTION-20250430T074713Z-001\NUMBER PLATE DETECTION\Automatic-License-Plate-Detection-main\Automatic-License-Plate-Detection-main\labels.csv')

path = glob(r'C:\Users\dhruv\Downloads\NUMBER PLATE DETECTION-20250430T074713Z-001\NUMBER PLATE DETECTION\Automatic-License-Plate-Detection-main\Automatic-License-Plate-Detection-main\images/*.xml')

#Converting xml file to csv and saving their bounding boxes coordinates
labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for filename in path:
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
df.to_csv(r'C:\Users\dhruv\Downloads\NUMBER PLATE DETECTION-20250430T074713Z-001\NUMBER PLATE DETECTION\Number_PlateRecognition\Number_PlateRecognition\labels.csv',index=False)
df.head()

filename = df['filepath'][0]
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join(r'C:\Users\dhruv\Downloads\NUMBER PLATE DETECTION-20250430T074713Z-001\NUMBER PLATE DETECTION\Number_PlateRecognition\Number_PlateRecognition\images',filename_image)
    return filepath_image
getFilename(filename)

image_path = list(df['filepath'].apply(getFilename))
image_path[:10]#random check

#Draw coordinates into rectangle form, so that we can verify that annotations are correct or not ?
file_path = image_path[8] #path of our image N2.jpeg
img = cv2.imread(file_path) #read the image
# xmin-1804/ymin-1734/xmax-2493/ymax-1882 
img = io.imread(file_path) #Read the image
fig = px.imshow(img)
fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 8 - N2.jpeg with bounding box')
fig.add_shape(type='rect',x0=1804, x1=2493, y0=1734, y1=1882, xref='x', yref='y',line_color='cyan')

labels = df.iloc[:,1:].values
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
    
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 # Normalization
    
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) # Normalized output
    # Append
    data.append(norm_load_image_arr)
    output.append(label_norm)

# Convert data to array
X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
# ---------------------
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)


# ---------- model
model = Model(inputs=inception_resnet.input,outputs=headmodel)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
model.summary()

tfb = TensorBoard('plate_detection')
history = model.fit(x=x_train, y=y_train, batch_size=10, epochs=100, validation_data=(x_test,y_test), callbacks=[tfb])
model.save('NumberPlate_detection1.h5')
model.save('NumberPlate_detection1.keras')

model_path = r'C:\Users\dhruv\Downloads\NUMBER PLATE DETECTION-20250430T074713Z-001\NUMBER PLATE DETECTION\NumberPlate_detection1.keras'
new_model = tf.keras.models.load_model(model_path)
new_model.summary()
print('Model Uploaded Successfully')

# Load and preprocess the image
image_path = r'C:\Users\dhruv\Downloads\NUMBER PLATE DETECTION-20250430T074713Z-001\NUMBER PLATE DETECTION\sample_images\N40.jpeg'
image = load_img(image_path)  # PIL object
image = np.array(image, dtype=np.uint8)  # 8-bit array (0,255)
image_resized = load_img(image_path, target_size=(224,224))
image_arr_test = img_to_array(image_resized) / 255.0  # Normalize to [0,1]

# Get original image size
h, w, d = image.shape
print('Height of the image =', h)
print('Width of the image =', w)

fig = px.imshow(image)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='TEST Image')
fig.show()

test_arr = image_arr_test.reshape(1, 224, 224, 3)
coords = new_model.predict(test_arr)

denorm = np.array([w, w, h, h])
coords = coords * denorm
coords = coords.astype(np.int32)

xmin, ymin, xmax, ymax = coords[0]
pt1 = (xmin, ymin)
pt2 = (xmax, ymax)
print(pt1, pt2)
cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)

# Display the image with bounding box
fig = px.imshow(image)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))
fig.show()