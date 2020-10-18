# Running TF object detection model in C# using onnxruntime


## Installation
Install **python 3+** dependencies.  
```
tensorflow: 1.12.0
onnx: 1.6.0
tf2onnx: 1.5.5
onnxruntime: 1.1.0
```

## TF to ONNX
Download TF object detection model from the [**Model Zoo**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and convert it with to onnx model.  
For example download [**ssd_mobilenet_v1_coco_2018_01_28**](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) and convert it to onnx model from **saved_model.pb**
```
python -m tf2onnx.convert 
  --opset 11 
  --fold_const 
  --saved-model ssd_mobilenet_v1_coco_2018_01_28/saved_model/ 
  --output ssd_mobilenet_v1_coco_2018_01_28/model.onnx
```
or from **frozen_inference_graph.pb**
```
python -m tf2onnx.convert 
  --opset 11 
  --fold_const 
  --graphdef ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb 
  --output ssd_mobilenet_v1_coco_2018_01_28/frozen.onnx 
  --inputs image_tensor:0 
  --outputs detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0
```
For running python and C# examples [**download**](https://yadi.sk/d/ap5F1VPma6qMGQ?w=1) path with already-made onnx models and move it to repository root folder.

## Python script
Run python script [**object_detection_image_onnx.py**](object_detection_image_onnx.py) to test converted onnx model.  
<p align="center"><img width="80%" src="docs/python.jpg" /></p>
<p align="center"><b>Figure 1.</b> Python example</p>  

## C# application
Define model and image paths
```c#
string model = @"...\ssd_mobilenet_v1_coco_2018_01_28\model.onnx";
string file = @"...\ssd2onnx\images\airport.jpg";
```
Build [**C#**](/csharp) source code and run application.  
<p align="center"><img width="80%" src="docs/csharp.jpg" /></p>
<p align="center"><b>Figure 2.</b> C# example</p>  

## References
[1] TensorFlow detection [**model zoo**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).  
[2] Tutorial: how to convert them to ONNX and run them under [**onnxruntime**](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb).  
[3] Microsoft: ONNX Runtime [**C#**](https://github.com/microsoft/onnxruntime/blob/master/docs/CSharp_API.md) API.  
[4] Ready-made ssd_mobile_net_v1 [**onnx-models**](https://yadi.sk/d/RQlhrFvJktxMpw).  
