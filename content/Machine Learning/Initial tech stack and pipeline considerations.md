---
title: Tech stack and pipeline
publish: true
---
### 1) Stack
- Runtime: Python
    
- Image IO / ops: opencv-python, Pillow, numpy
    
- Object detection:
    
    - Car mask: Ultralytics YOLOv8-seg (n or s) pre-trained (no training needed).
        
    - Plate box: Ultralytics YOLOv8 (n or s) fine-tuned on plates (small dataset).
        
- OCR: EasyOCR (v1), with whitelist and single-line mode. Keep Tesseract as a fallback only if you need a pure-offline path on low-end hardware.
    
- API: Flask (simple) or FastAPI (nicer typing). Pick one; I’ll assume Flask.
    
- Optional later: ONNX Runtime for inference speed/CPU, or TensorRT if you get a CUDA box.