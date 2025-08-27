---
title: Potential Tech Stack
publish: true
---
**Plate Detection:**
- Ultralytics (YOLOv8)
- Or cvlib / OpenCV with custom Haar or YOLO models

**OCR:**
- EasyOCR
- pytesseract (Python wrapper for Tesseract)

**Image Handling / Preprocessing:**
- opencv-python (cv2)
- Pillow

**Color Analysis:**
- skimage or raw NumPy over pixel arrays
- Use `cv2.cvtColor(..., cv2.COLOR_BGR2HSV)` and bin values into color buckets

**Optional GUI / API / Control:**
- Flask for API
- Tkinter / PyQT / Streamlit for basic GUI if needed