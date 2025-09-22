![[Pasted image 20250909191616.png]]

![[Pasted image 20250909193753.png]]


(.venv) jbs@jbs:~/venvs/PaddleOCR$ python tools/export_model.py   -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml   -o Global.pretrained_model=./en_PP-OCRv5_mobile_rec_pretrained.pdparams      Global.save_inference_dir=./inference/en_ppocrv5_mobile_rec_clean
/home/jbs/venvs/.venv/lib/python3.12/site-packages/paddle/utils/cpp_extension/extension_utils.py:718: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
Skipping import of the encryption module.
[2025/09/12 23:43:24] ppocr INFO: load pretrain successful from ./en_PP-OCRv5_mobile_rec_pretrained
[2025/09/12 23:43:24] ppocr INFO: Export inference config file to ./inference/en_ppocrv5_mobile_rec_clean/inference.yml
Skipping import of the encryption module
W0912 23:43:25.144865 450274 eager_utils.cc:3441] Paddle static graph(PIR) not support input out tensor for now!!!!!
[2025/09/12 23:43:27] ppocr INFO: inference model is saved to ./inference/en_ppocrv5_mobile_rec_clean/inference

(.venv) jbs@jbs:~/venvs/PaddleOCR$ paddle2onnx   --model_dir ./inference/en_ppocrv5_mobile_rec_clean   --model_filename inference.json   --params_filename inference.pdiparams   --save_file ./en_ppocrv5_mobile_rec_clean.onnx   --opset_version 11
/home/jbs/venvs/.venv/lib/python3.12/site-packages/paddle/utils/cpp_extension/extension_utils.py:718: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
[Paddle2ONNX] Start parsing the Paddle model file...
[Paddle2ONNX] Use opset_version = 14 for ONNX export.
[Paddle2ONNX] PaddlePaddle model is exported as ONNX format now.
2025-09-12 23:43:33 [INFO]	Try to perform optimization on the ONNX model with onnxoptimizer.
2025-09-12 23:43:33 [INFO]	ONNX model saved in ./en_ppocrv5_mobile_rec_clean.onnx.

Transformation til onnx - hvorfor? Superirriterende før - kunne have holdt den kørende dog. Men ville stadig ikke fungere til mobil.

venv

Stanford resume?




`--dynamic-export` is an optional flag that makes the ONNX graph **accept variable input sizes** instead of a fixed 640×640.

- Without it: the exported model will only accept exactly 640×640 images.
    
- With it: the ONNX graph keeps symbolic dimensions so you can feed arbitrary H×W at runtime.  
    For mobile deployment this is handy if you plan to resize inputs differently per device.
    

The main trade-off is runtime efficiency:

- **Static export** (fixed 640×640) lets the backend pre-optimize tensor shapes. ONNX Runtime, TensorRT, CoreML etc. can fuse ops and allocate memory once. Inference is usually a bit faster and lighter.
    
- **Dynamic export** adds symbolic dimensions. The engine has to handle arbitrary shapes, so some graph optimizations and kernel fusions can’t be hard-coded. Slightly slower and a bit more RAM overhead.
    

If your app will always resize to a single size (e.g. always 640×640), stick with static.  
If you expect variable camera inputs or want flexibility for future retraining, dynamic is worth it—the slowdown is usually small compared to the convenience.





ONNX ISSUE

The Core Problem
The initial issue was that the detect_vehicle_rtm_onnx function was not correctly processing the output from the ONNX model. The model was fed an image that was resized and padded to fit a 640x640 canvas (a technique called letterboxing). Consequently, the model's output—both bounding boxes and segmentation masks—was in the coordinate system of that 640x640 canvas, not the original image.

The original code and my first few attempts failed to correctly translate these coordinates and masks back to the original image space. This resulted in either an incorrect, large crop (as you observed) or an empty crop, which caused the downstream cv2.resize error.

The Failed Attempts
Dynamic Sizing: We tried removing the 640x640 resizing. This failed because the model's architecture requires input dimensions to be divisible by a certain number (usually 32). Arbitrary image sizes caused a dimension mismatch error inside the ONNX model's network layers.
Padding to 32: We then tried padding the image to the nearest multiple of 32. While this satisfied the model's input requirement, it was memory-unsafe. A large input image would be padded to an even larger size, consuming excessive RAM and nearly crashing your computer.
The Final Solution (The Trick)
The final, working solution returned to the stable and memory-safe letterboxing approach but implemented the post-processing correctly. This was the "trick": a precise, multi-step reversal of the input transformation.

Input: The image is resized to fit in a 640x640 box, noting the scale factor and the top/left padding added.

Output: The model gives a low-resolution mask and a bounding box for the 640x640 canvas.

Correct Mask Transformation:

The low-resolution mask is first resized to the full canvas size (640x640).
We then "un-pad" it by cropping the mask to the area where the resized image was placed (mask_canvas[top:top+nh, left:left+nw]).
Finally, this cropped mask is resized back to the original image's dimensions (w0, h0). This gives a pixel-perfect mask for the original image.
Correct Bounding Box Transformation (for the fallback case):

The bounding box coordinates from the model are first adjusted for padding by subtracting left and top.
Then, they are scaled back to the original size by dividing by the scale factor.
By correctly reversing every step of the input preprocessing, we could accurately map the model's detection onto the original image, producing a tight, correct crop. This valid crop allowed the rest of your pipeline to execute successfully.