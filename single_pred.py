import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from Utils import decode, yolo_loss_func, load_model, load_date, OUTPUT_FOLDER_PATH, optimizer, DATA_PATH

# Variable Definition

img_w = 512
img_h = 512
channels = 3
classes = 1
info = 5
grid_w = 16
grid_h = 16

# optimizer


def predict_func(model, inp, iou, name):
    ans = model.predict(inp)

    # np.save('Results/ans.npy',ans)
    boxes = decode(ans[0], img_w, img_h, iou)

    img = ((inp + 1) / 2)
    img = img[0]
    # plt.imshow(img)
    # plt.show()

    for i in boxes:
        i = [int(x) for x in i]

        img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), color=(0, 255, 0), thickness=2)

    plt.imshow(img)
    plt.show()

    cv2.imwrite(os.path.join(OUTPUT_FOLDER_PATH, str(name) + '.jpg'), img * 255.0)


model = load_model('model/text_detect_model.json')
model.load_weights('model/text_detect.h5')

model.compile(loss=yolo_loss_func, optimizer=optimizer, metrics=['accuracy'])

# import data
# X and Y numpy arrays are created using the Prepocess.py file
X, Y = load_date(data_path=DATA_PATH)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.75, shuffle=True)
X = []
Y = []

rand = np.random.randint(0, X_val.shape[0], size=5)

for i in rand:
    predict_func(model, X_val[i:i + 1], 0.5, i)
