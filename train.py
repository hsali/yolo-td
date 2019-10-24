from Utils import save_model, yolo_model, img_h, img_w, channels, MODEL_PATH


input_size = (img_h, img_w, channels)
model = yolo_model(input_size)

save_model(model, model_path=MODEL_PATH)
