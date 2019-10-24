from sklearn.model_selection import train_test_split

from Utils import yolo_loss_func, load_model, load_date, optimizer, DATA_PATH, MODEL_PATH, accuracy_checking

model = load_model(MODEL_PATH)
model.compile(loss=yolo_loss_func, optimizer=optimizer, metrics=['accuracy'])

X, Y = load_date(data_path=DATA_PATH)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.75, shuffle=False)

accrucay = accuracy_checking(model, X, Y)
print("model accuracy is : {}".format(accrucay))