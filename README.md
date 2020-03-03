# Running object detection model in C# using onnxruntime


## Installation
For this tutorial use used the following versions:
```
python 3.7
tensorflow: 1.13.1
onnx: 1.5.1
tf2onnx: 1.5.1
onnxruntime: 0.4
```

## Preparing
Download [**ssd_mobilenet_v1_coco**](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) and move to the root folder.  
Run **ssd2onnx.bat** to convert **saved_model.pb** to **model.onnx**.  
Run [**object_detection_image_onnx.py**](object_detection_image_onnx.py) to test saved onnx model.  
Build [**csharp**](/csharp) source code and run application.  

## References
[1] Tutorial: how to convert them to ONNX and run them under [onnxruntime](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb). 

