from yolov5 import YOLOv5

# set model params
model_path = "yolov5/weights/yolov5s.pt" # it automatically downloads yolov5s model to given path
device = "cpu" # "cuda" or "cpu"

# init yolov5 model
yolov5 = YOLOv5(model_path, device)

# load images
image1 = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
image2 = 'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg'

# perform inference
results = yolov5.predict(image1)

# perform inference with larger input size
results = yolov5.predict(image1, size=1280)

# perform inference with test time augmentation
results = yolov5.predict(image1, augment=True)

# perform inference on multiple images
results = yolov5.predict([image1, image2], size=1280, augment=True)

# show detection bounding boxes on image
results.show()

# save results into "results/" folder
results.save(save_dir='results/')