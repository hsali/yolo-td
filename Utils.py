import os
import cv2
import matplotlib.pyplot as plt
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import *
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import Adam

# output dims -> (1,x,x,1,5)

# boxes = decode_to_boxes(output)  output to boxes
# corner_boxes = boxes_to_corners(boxes) boxes to corners
# final_out = non_max_suppress(corner_boxes) 
#                   iou()

OUTPUT_FOLDER_PATH = "Results2"
MODEL_PATH = "model"
DATA_PATH = "Data"
IOU = 0.5

# Variable Definition
img_w = 512
img_h = 512
channels = 3
classes = 1
info = 5
grid_w = 16
grid_h = 16

# optimizer
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


def load_date(data_path='Data'):
    """
    load data
    Parameters
    ----------
    data_path

    Returns
    -------

    """
    X = np.load(os.path.join(data_path, 'X.npy'))
    Y = np.load(os.path.join(data_path, 'Y.npy'))
    return X, Y


def yolo_model(input_shape):
    """
    define model
    # input : 512,512,3
    # output : 16,16,1,5

    Parameters
    ----------
    input_shape

    Returns
    -------

    """
    inp = Input(input_shape)

    model = MobileNetV2(input_tensor=inp, include_top=False, weights='imagenet')
    last_layer = model.output

    conv = Conv2D(512, (3, 3), activation='relu', padding='same')(last_layer)
    conv = Dropout(0.4)(conv)
    bn = BatchNormalization()(conv)
    lr = LeakyReLU(alpha=0.1)(bn)

    conv = Conv2D(128, (3, 3), activation='relu', padding='same')(lr)
    conv = Dropout(0.4)(conv)
    bn = BatchNormalization()(conv)
    lr = LeakyReLU(alpha=0.1)(bn)

    conv = Conv2D(5, (3, 3), activation='relu', padding='same')(lr)

    final = Reshape((grid_h, grid_w, classes, info))(conv)

    model = Model(inp, final)

    return model


# define loss function
def yolo_loss_func(y_true, y_pred):
    """
    yolo loss functions

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    """
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


def save_model(model, model_path="model1"):
    """
    save model
    Parameters
    ----------
    model
    model_path

    Returns
    -------

    """
    model_json = model.to_json()
    with open(os.path.join(model_path, "text_detect_model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(model_path, 'text_detect.h5'))


def load_model(model_path):
    """
    load path
    Parameters
    ----------
    model_path

    Returns
    -------
    return loaded_model
    """
    json_file = open(os.path.join(model_path, 'text_detect_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(model_path, 'text_detect.h5'))
    return loaded_model


def decode_to_boxes(output, ht, wd):
    # output : (x,x,1,5)
    # x,y,h,w

    img_ht = ht
    img_wd = wd
    threshold = 0.5
    grid_h, grid_w = output.shape[:2]
    final_boxes = []
    scores = []

    for i in range(grid_h):
        for j in range(grid_w):
            if output[i, j, 0, 0] > threshold:
                temp = output[i, j, 0, 1:5]

                x_unit = ((j + (temp[0])) / grid_w) * img_wd
                y_unit = ((i + (temp[1])) / grid_h) * img_ht
                width = temp[2] * img_wd * 1.3
                height = temp[3] * img_ht * 1.3

                final_boxes.append([x_unit - width / 2, y_unit - height / 2, x_unit + width / 2, y_unit + height / 2])
                scores.append(output[i, j, 0, 0])

    return final_boxes, scores


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    fin_area = area1 + area2 - inter

    iou = inter / fin_area

    return iou


def non_max(boxes, scores, iou_num):
    scores_sort = scores.argsort().tolist()
    keep = []

    while len(scores_sort):

        index = scores_sort.pop()
        keep.append(index)

        if (len(scores_sort) == 0):
            break

        iou_res = []

        for i in scores_sort:
            iou_res.append(iou(boxes[index], boxes[i]))

        iou_res = np.array(iou_res)
        filtered_indexes = set((iou_res > iou_num).nonzero()[0])

        scores_sort = [v for (i, v) in enumerate(scores_sort) if i not in filtered_indexes]

    final = []

    for i in keep:
        final.append(boxes[i])

    return final


def decode(output, ht, wd, iou):
    boxes, scores = decode_to_boxes(output, ht, wd)
    boxes = non_max(boxes, np.array(scores), iou)
    return boxes


def predict(model, input):
    """
    predict images
    Parameters
    ----------
    model
    input

    Returns
    -------

    """
    ans = model.predict(input)
    img = ((input + 1) / 2)
    img = img[0]
    return ans, img


def draw_rectangle(img, rect):
    """
    draw rectangle
    Parameters
    ----------
    img
    rect

    Returns
    -------

    """
    return cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 255, 0), thickness=2)


def save_image(img, name):
    """
    save image
    Parameters
    ----------
    name
    img

    Returns
    -------

    """
    cv2.imwrite(os.path.join(OUTPUT_FOLDER_PATH, str(name) + '.jpg'), img * 255.0)


def predicted_box(predicted_output):
    """
    convert predicted output to rectangle
    Parameters
    ----------
    predicted_output

    Returns
    -------

    """
    boxes = decode(predicted_output[0], img_w, img_h, IOU)
    rectangles = []
    for r in boxes:
        rectangles.append([int(p) for p in r])
    return rectangles


def show_image(img):
    """
    plot image
    Parameters
    ----------
    img

    Returns
    -------

    """
    plt.imshow(img)
    plt.show()


def accuracy_checking(model, X, Y):
    data_checks = []
    for i in range(X.shape[0]):
        print("validating and checking: {}/{}".format(i, X.shape[0]))
        pred_out, img = predict(model, X[i:i + 1])
        rects = predicted_box(pred_out)
        pred_rect_len = len(rects)
        act_rects = predicted_box(Y[i:i + 1])
        act_rects_len = len(act_rects)
        data_checks.append([act_rects_len, pred_rect_len, act_rects_len == pred_rect_len])

    accuracy = sum([1 for x in data_checks if x[2] == True]) / len(data_checks) * 100.00

    return accuracy


def predict_func(model, inp, name, image_save=False):
    """
    predict image and show and save
    Parameters
    ----------
    model
    inp
    name
    image_save

    Returns
    -------

    """
    r_img = None
    pred_out, img = predict(model, inp)

    rects = predicted_box(pred_out)
    for rect in rects:
        r_img = draw_rectangle(img, rect)
    print('draw rectangle image')
    show_image(r_img)
    if image_save:
        save_image(r_img, name)