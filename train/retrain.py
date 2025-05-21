from ultralytics import YOLO

DATASET_YAML = "/home/minghao/Documents/Gits/OutdoorSensorNodes/CoInfra/test_code/OutdoorData.yaml"
# FineTune the model
model = YOLO('yolo11s-obb.pt')
# model = YOLO('runs/obb/train2/weights/best.pt')

# results = model.train(data='OutdoorData.yaml', epochs=100)

results = model.train(data=DATASET_YAML, epochs=30, scale=0.0, shear=0.0, degrees=45.0, fliplr=0.0, plots=True, close_mosaic=10,
                      hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, perspective=0.0, project="late_fusion")