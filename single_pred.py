from sklearn.model_selection import train_test_split
from Utils import yolo_loss_func, load_model, load_date, optimizer, DATA_PATH, predict, predicted_box, draw_rectangle, \
    show_image, save_image, MODEL_PATH


def predict_func(model, inp, name, image_save=False):
    r_img  = None
    pred_out, img = predict(model, inp)

    rects = predicted_box(pred_out)
    for rect in rects:
        r_img = draw_rectangle(img, rect)
    print('draw rectangle image')
    show_image(r_img)
    if image_save:
        save_image(r_img, name)


model = load_model(MODEL_PATH)
model.compile(loss=yolo_loss_func, optimizer=optimizer, metrics=['accuracy'])

# import data
# X and Y numpy arrays are created using the Prepocess.py file
X, Y = load_date(data_path=DATA_PATH)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.75, shuffle=False)
X = []
Y = []

# show_image(X_val[0])
predict_func(model, X_val[0:1], 'img_0')
