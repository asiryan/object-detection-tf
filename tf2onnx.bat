@echo off
python -m tf2onnx.convert --opset 11 --fold_const --saved-model ssd_mobilenet_v1_coco_2018_01_28/saved_model/ --output ssd_mobilenet_v1_coco_2018_01_28/model.onnx
rem python -m tf2onnx.convert --opset 11 --fold_const --graphdef ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb --output ssd_mobilenet_v1_coco_2018_01_28/frozen.onnx --inputs image_tensor:0 --outputs detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0
pause