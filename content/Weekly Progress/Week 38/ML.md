---
title: ML
publish: true
draft: false
---
### Last week
Finished Python Guide. Started experimenting with python code and computer vision models.

### Short-Term Plan
- Stop being sick
- If possible experiment with different project-relevant models.

### Process
Experimented with a subsystem for recognizing color. Based on HSV-dominant pixel values in a complete cropped image of a car. Played around with different thresholds for categorizing colors, different brightness optimization approaches, different ways of factoring brightness values for accurate readings. In the end, results were decent for the test images - I'm aware that this doesn't necessarily generalize well:
![[Pasted image 20250922220128.png]]

This slide is from a lecture from the Stanford University course - even though this is about hyperparameters, the point about not relying on test data for validation also holds in this case. 

```python
def detect_color(image, mask):

    # ── 1. Validate mask ───────────────────────────────────────────────
    if mask is None or np.count_nonzero(mask) == 0:
        return "unknown"

    # ── 2. Optional spatial crop ───────────────────────────────────────
    # Optional: chop off top 30% of image + mask to ignore roof reflections etc.

    chop_top = 0.3
    chop_sides = 0.15
    chop_bottom = 0.2

    h, w = image.shape[:2]
    top = int(h * chop_top)
    bottom = int(h * (1 - chop_bottom))
    left = int(w * chop_sides)
    right = int(w * (1 - chop_sides))

    image = image[top:bottom, left:right]
    mask = mask[top:bottom, left:right]
    cv2.imwrite(unique_filename("processed/CDcrop.jpg"), image)
    cv2.imwrite(unique_filename("processed/CDMaskcrop.jpg"), mask)

    # ── 3. HSV conversion + masking ────────────────────────────────────
    # Ensure mask is boolean
    mask = mask > 0

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply mask to each channel
    h = h[mask]
    s = s[mask]
    v = v[mask]

    # ── 4. pixel classification ────────────────────────────────────────
    total = s.size
    black_pixels = np.sum(v < 50)
    gray_pixels = np.sum((v >= 50) & (v < 200) & (s < 40))
    white_pixels = np.sum((v >= 200) & (s < 40))
    color_pixels = np.sum((s >= 40) & (v >= 40))

    print(f"""
White pixels: {round(white_pixels/total*100)}
Black pixels: {round(black_pixels/total*100)}
Gray pixels: {round(gray_pixels/total*100)}
Color pixels: {round(color_pixels/total*100)}""")

    # ── 5. neutral color heuristic ─────────────────────────────────────
    # Kinda white hack (the honda fix)
    neutral_colors = {
        "white": white_pixels,
        "gray": gray_pixels,
        "black": black_pixels,
    }
    dominant_neutral = max(neutral_colors, key=neutral_colors.get)
    dominant_value = neutral_colors[dominant_neutral]

    if (white_pixels > total * 0.25):
        return "white"

    if (
        dominant_value > 0.4 * total
        or (dominant_value > 0.35 * total and color_pixels < 0.65 * total)
        or (dominant_neutral == "white" and dominant_value > 0.125 * total and white_pixels > black_pixels + gray_pixels)
    ):
        return dominant_neutral

    if color_pixels < 0.4 * total and dominant_value > 0.2 * total:
        return dominant_neutral

    # ── 6. hue-based color categorization ──────────────────────────────
    hue = h[(s > 40) & (v > 40)]
    if len(hue) == 0:
        return "unknown"

    hist, _ = np.histogram(hue, bins=180, range=(0, 180))
    mode_hue = int(np.argmax(hist))

    print(f"""Mode_hue (for color categorization): {round(mode_hue)}
""")

    if mode_hue < 10 or mode_hue > 160:
        return "red"
    elif mode_hue < 22:
        return "orange"
    elif mode_hue < 32:
        return "yellow"
    elif mode_hue < 45:
        return "lime"
    elif mode_hue < 85:
        return "green"
    elif mode_hue < 125:
        return "blue"
    elif mode_hue < 160:
        return "purple"
    else:
        return "unknown"
```

Also experimented with object detection models (YOLO, mmdetection) and optical character recognition (OCR) models (PaddleOCR). Got them working and exported to onnx format, which is possible to get running on mobile phones, which is necessary for our project. This was mostly proof of concept work to find out if we could get the models up and running in an environment that was transferable to phones.

### Results

Original image:
![[car1 1.png]]

Brightness adjusted and cropped based on car recognition:
![[crop.jpg]]

Car mask:
![[crop-mask.jpg]]

Further zoom for color detection:
![[CDcrop.jpg]]

Zoomed mask, to avoid reading color outside the car area (maybe not the best example, but the top right corner shows the logic nonetheless - the black pixels are not part of the car, so color detection will not include these pixels. This prevents background color from being included when detecting color):
![[CDMaskcrop.jpg]]

Grayscaled license plate to avoid background color noise and improve contrast between characters and non-characters. 
![[plate-crop.jpg]]

Results from running the whole pipeline. Both license plate and color is correctly detected:
![[Pasted image 20250922220951.png]]
### Reflections
We managed to get a proof of concept version working. This showed that our approach was realistic and worth using going forwards. However for the proof of concept work we got quite a bit of help from LLMs both in exporting models to onnx format and getting the models running correctly. I don't see a big problem with getting help exporting the models, since the result of this is boolean - either it works or it doesn't - so I'm not hugely concerned with understanding every single part of this process completely in depth. Obviously we should have a general idea of what is happening - but again - this is about going from one runtime to another without changing the underlying logic of the model. So there isn't a lot to make sense of here - just reading documentation enough to make the 'translation' work.

I'm more interested in the code that actually uses the models. This was also written with help from LLMs - again because it was important to prove the concept - was this approach realistic and worth using going forward. Since we got it working, the approach does make sense, and now we will work on breaking down the code. We agree in the ML-team that we need to understand every single line of code we might send into production at some point, so this will be a big part of our work for the first MVP.

Another hiccup we encountered was the fact that some of the models we have been experimenting with fall under licenses that restrict us in some ways. We have therefor been looking into alternatives and got an alternative model working for recognizing cars in images. The only piece of the puzzle still relying on restricted licenses is the licens plate recognition. Because of this we are looking into finetuning the model we are currently using for car recognition to also be able to recognize license plates. This doesn't seem unrealistic to accomplish, but we need to further investigate this matter.

Finally we also learned that Apple doesn't allow Python code to run on their phones. This is a pretty big problem for us, and we are looking into ways to get around this problem.

### Coming week

The Python-Iphone problem has taken first priority right now, because we need to solve this problem in order to get a MVP in the hands of users for testing. The models we currently have working are not restricted for user testing, so we can use the current setup for user feedback even though these are not yet optimized or necessarily perfectly implemented yet. So even though they are not production ready, the current setup might be good enough for user testing. Internal on device testing will be done before user tests.