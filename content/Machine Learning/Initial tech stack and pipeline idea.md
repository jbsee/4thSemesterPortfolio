---
title: Initial tech stack and pipeline idea
publish: true
---
## 1) Stack
#### Runtime
- Python
#### Image IO / ops
<details> <summary><strong>Pillow</strong> to open the raw photo correctly.</summary>Pillow is a general-purpose image library. It’s better than cv2 at things like opening phone photos (EXIF orientation correction, weird formats), saving to different formats, quick manipulations. <strong>Often you open with Pillow, then convert to numpy arrays for OpenCV.</strong><br><br>
	</details> 

<details> <summary>Convert to a <strong>numpy array.</strong></summary>Numpy is the foundation. Both cv2 and Pillow hand you image data as numpy arrays (height × width × channels). You’ll use numpy operations directly when you need to average pixels, build histograms, mask out regions, etc.<br><br>
	</details> 

<details> <summary>Use <strong>OpenCV</strong> for all the vision-specific operations.</summary>Swiss army knife for computer vision. Load images, resize, crop, warp, draw boxes, convert color spaces (RGB → HSV), blur, threshold, etc.
	</details> 

#### Object detection
- Car mask: Ultralytics yolov8-seg.pt (n or s) pre-trained.
- License plate: Ultralytics yolov8-??? (n or s) pre-trained.
      <details> <summary><strong>YOLOv8 by Ultralytics</strong></summary>
      <table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Size</th>
      <th>Speed</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>n</code></td>
      <td>nano</td>
      <td><strong>Fastest</strong></td>
      <td>Lowest</td>
    </tr>
    <tr>
      <td><code>s</code></td>
      <td>small</td>
      <td>Fast</td>
      <td>Good</td>
    </tr>
    <tr>
      <td><code>m</code></td>
      <td>medium</td>
      <td>Slower</td>
      <td>Better</td>
    </tr>
    <tr>
      <td><code>l</code></td>
      <td>large</td>
      <td>Slower</td>
      <td>Even better</td>
    </tr>
    <tr>
      <td><code>x</code></td>
      <td>extra large</td>
      <td><strong>Slowest</strong></td>
      <td>Best</td>
    </tr>
  </tbody>
	   </table>
	   <ul>
	   <li>
	   Nano for fast, lightweight inference (e.g. mobile, CPU, quick dev).
	   </li>
	   <li>
	   Small for better accuracy than nano (e.g. PC with decent performance).
	   </li>
	   </ul>
	   <br>Both nano and small are good starting points for real-time-ish use. Try nano first, bump to small only if accuracy is too low.
	   <br><br>
	</details> 
	<details> <summary><strong>Pretrained vs. finetuned</strong></summary>
	<ul>
	   <li>Try a <strong>pretrained YOLOv8 model for plates</strong> - e.g. available on HuggingFace or Roboflow.
	   </li>
	   <li> <strong>Test the whole pipeline end-to-end:</strong> <br>(detect → crop → OCR → color).
	   </li>
	   <li>If detection is consistently failing (wrong boxes, missing plates), <strong>then</strong> you:
	   <ul>
	   <li>Collect ~500+ labeled photos from your region.</li>
	   <li>Fine-tune YOLOv8 on just that class (`license_plate`).</li>
	   <li>Swap out the detector model.</li>
	   </ul> </li> </ul><br>
	   </details>

#### Optical Character Recognition (OCR)
<details> <summary><strong>EasyOCR</strong> with whitelist and single-line mode.</summary>Python OCR-library.</strong>
	<ul>
	   <li>
	   <strong>Whitelist:</strong> Allowed characters. Model ignores irrelevant characters.
	   </li>
	   <li>
	   <strong>Single-line mode:</strong> EasyOCR has a both a paragraph and a line inference mode. Setting it to single-line avoids weird layout parsing and improves results. 
	   </li>
	   </ul>
</details> 
<details> <summary><strong>Tesseract</strong> as fallback.</summary>Tesseract is an older alternative model that can be used as a fallback if we need a pure-offline path on low-end hardware.<br><br>
	</details>  
    
#### API
- **Flask** (simple) or **FastAPI** (nicer typing).
- Potential optimization:
	-  By default, YOLOv8 runs with **PyTorch**. It's simple and flexible, but not the fastest - especially on CPU.
  <details> <summary><strong>ONNX</strong> Runtime for inference speed/CPU.</summary>	   
  <ul>
	   <li>
	   A fast, portable inference engine.
	   </li>
	   <li>
	   You convert your YOLOv8 model to .onnx, and then run it with ONNX Runtime.
	   </li>
	   <li>
	   It runs faster on CPU and supports some GPU acceleration too.
	   </li>
	   <li>
	   Great if you’re deploying on machines without GPUs but still want speed.
	   </li>
	   </ul><br>
	</details> 
  <details> <summary><strong>TensorRT</strong> for CUDA.</summary>  <ul>
	   <li>
	   NVIDIA’s ultra-optimized inference engine for GPU.
	   </li>
	   <li>
	   Converts your model to a TensorRT engine and runs it lightning-fast.
	   </li>
	   <li>
	   Useful if you deploy on a CUDA-enabled GPU (e.g., Jetson, RTX cards).
	   </li>
	   <li>
	   Setup is trickier, and it’s NVIDIA-only.
	   </li>
	   </ul><br>
	</details> 

## 2) End-to-end pipeline
#### 1) Load image
- Fix EXIF orientation
- Downscale to a max side.
#### 2) Plate detection
- Run YOLOv8 plate detector
	- Choose boxes that intersect the car mask.
	- If multiple, keep the highest confidence.
    
#### 3) Plate crop preprocessing
- Perspective rectify (four-point warp if needed)
- Convert to grayscale
- Light denoise
- Adaptive threshold - keep an un-thresholded copy as OCR fallback.

#### 4) OCR
- EasyOCR with Latin, `allowlist = 0-9A-Z`, single-line mode.
- Post-process
	- Strip spaces and punctuation.
	- Regex sanity filter (A-Z, 0-9, length 5–8), reject if confidence < threshold.
        
#### 5) Car color
- Compute HSV histogram excluding low-saturation (windows/gray/black) and extreme V (near-black/near-white). Take dominant hue bucket; map to label set:
    - {white, silver, gray, black, red, orange, yellow, green, blue, purple, brown}  
        Heuristics:
    - “White”: V high, S very low
    - “Black”: V very low
    - “Silver/Gray”: S low, V mid/high (threshold separates silver vs gray by V)
        
#### 6) Output JSON
- plate_text
- plate_conf
- color_label 
- color_conf
- boxes/mask (for debugging)
- overall quality flag

## 3) Potential fine tuning
- Data: mix of your own phone photos + a public license-plate dataset; prioritize varied angles, day/night, occlusions.
- Label tool: Roboflow or Label Studio (just plate boxes).
- Train: Ultralytics CLI, 10–50 epochs on `yolov8n` or `yolov8s`. Early stop by val mAP.
- Export best weights; keep your YAML classes = {license_plate}.

## 4) Potential color mapping 
- Compute masked HSV histogram with S > 0.2 and 0.1 < V < 0.95.
- If S < 0.18 overall
	- If V > 0.85 → white
	- If 0.55 < V ≤ 0.85 → silver
	- Else → gray/black by V threshold
- Else use hue peaks:
    - 0±15/180±15 → red
    - 15–30 → orange
    - 30–60 → yellow
    - 60–150 → green → blue split at 105
    - 150–165 → purple
    - 165–180 and very low S → gray bucket guard
	    - Return label + share of pixels behind that decision as confidence.

## 5) Acceptance criteria
- Plate detector: mAP@0.5 ≥ 0.9 on your val set; precision ≥ 0.95 to avoid false positives.
- OCR: ≥ 95% exact-string accuracy on your held-out photos; otherwise keep a retry path (second crop variant, different binarization).
- Color: ≥ 90% agreement with human labels on 200 photos; review confusion between silver/gray/white.

## 6) Known edge cases
- Night shots and glare: add CLAHE before thresholding; consider second OCR pass on inverted image.
- Two-line plates: try a taller crop; regex with a join rule.
- Multiple cars: choose the car with the largest plate box overlap; otherwise largest car mask.
- Motion blur: short-circuit with a blur detector; tell the user to reshoot.
- European fonts with similar glyphs: map O↔0, I↔1, B↔8 cautiously; only apply if it increases plate pattern validity without lowering OCR confidence below threshold.