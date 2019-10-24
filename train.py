import os

from keras.applications.mobilenetv2 import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from Utils import save_model, yolo_model, load_date, img_h, img_w, channels, DATA_PATH, MODEL_PATH

# import data
# X and Y numpy arrays are created using the Prepocess.py file
X, Y = load_date(DATA_PATH)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.75, shuffle=True)
X = []
Y = []

input_size = (img_h, img_w, channels)
model = yolo_model(input_size)

print(model.summary())
save_model(model, model_path=MODEL_PATH)
