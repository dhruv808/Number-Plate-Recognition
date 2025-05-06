import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Set the environment variable
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reconfigure the stdout to use UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load the model
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

# Display the original image
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