from keras.engine.saving import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from Utils import *

import numpy as np


#Variable Definition
img_w = 512
img_h = 512
channels = 3
classes = 1
info = 5
grid_w = 16
grid_h = 16



# optimizer
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


def yolo_loss_func(y_true, y_pred):
    # y_true : 16,16,1,5
    # y_pred : 16,16,1,5
    l_coords = 5.0
    l_noob = 0.5
    coords = y_true[:, :, :, :, 0] * l_coords
    noobs = (-1 * (y_true[:, :, :, :, 0] - 1) * l_noob)
    p_pred = y_pred[:, :, :, :, 0]
    p_true = y_true[:, :, :, :, 0]
    x_true = y_true[:, :, :, :, 1]
    x_pred = y_pred[:, :, :, :, 1]
    yy_true = y_true[:, :, :, :, 2]
    yy_pred = y_pred[:, :, :, :, 2]
    w_true = y_true[:, :, :, :, 3]
    w_pred = y_pred[:, :, :, :, 3]
    h_true = y_true[:, :, :, :, 4]
    h_pred = y_pred[:, :, :, :, 4]

    p_loss_absent = K.sum(K.square(p_pred - p_true) * noobs)
    p_loss_present = K.sum(K.square(p_pred - p_true))
    x_loss = K.sum(K.square(x_pred - x_true) * coords)
    yy_loss = K.sum(K.square(yy_pred - yy_true) * coords)
    xy_loss = x_loss + yy_loss
    w_loss = K.sum(K.square(K.sqrt(w_pred) - K.sqrt(w_true)) * coords)
    h_loss = K.sum(K.square(K.sqrt(h_pred) - K.sqrt(h_true)) * coords)
    wh_loss = w_loss + h_loss

    loss = p_loss_absent + p_loss_present + xy_loss + wh_loss

    return loss


def load_model(strr):
    json_file = open(strr, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


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

    cv2.imwrite(os.path.join('Results2', str(name) + '.jpg'), img * 255.0)


model = load_model('model/text_detect_model.json')
model.load_weights('model/text_detect.h5')

model.compile(loss=yolo_loss_func, optimizer=opt, metrics=['accuracy'])






#import data
#X and Y numpy arrays are created using the Prepocess.py file
X = np.load('Data/X.npy')
Y = np.load('Data/Y.npy')


X_train , X_val , Y_train , Y_val  = train_test_split(X,Y,train_size = 0.75 , shuffle = True)
X = []
Y = []

rand = np.random.randint(0, X_val.shape[0], size=5)

for i in rand:
    predict_func(model, X_val[i:i + 1], 0.5, i)
