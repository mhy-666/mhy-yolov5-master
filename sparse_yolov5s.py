from sparsezoo.models import Zoo

stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96"
model = Zoo.download_model_from_stub(stub, override_parent_path="downloads")

# Prints the download path of the model
print(model.dir_path)