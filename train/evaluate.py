from ultralytics import YOLO

DATASET_YAML = "/home/minghao/Documents/Gits/OutdoorSensorNodes/CoInfra/test_code/OutdoorData.yaml"
# FineTune the model
# model = YOLO('yolo11s-obb.pt')
model = YOLO(
    '/home/minghao/Documents/Gits/OutdoorSensorNodes/CoInfra/late_fusion/train/weights/best.pt')

# results = model.train(data='OutdoorData.yaml', epochs=100)

results = model.val(data=DATASET_YAML, conf=0.25, iou=0.5, project="late_fusion",
                    name="eval", plots=True, split='test', save_json=True)
