# OpenCV Computer Vision Playground

A hands-on repository exploring core Computer Vision concepts and fundamental image processing pipelines using OpenCV and Python.

## 🚀 Features & Pipeline Covered

This workspace implements the classical Computer Vision pipeline:

1. **Video Streaming & Webcam**: Real-time webcam frame acquisition, ASCII bitmasking, and safe resource release (`video.py`).
2. **Color Spaces**: BGR to Grayscale, RGB, and HSV conversions (`image.py`).
3. **Image Smoothing & Noise Reduction**: Gaussian Blur, Median Blur, and Average Blur.
4. **Binarization & Thresholding**:
   - Simple Thresholding (Fixed midpoint)
   - Otsu's Thresholding (Global automatic bimodal optimization)
   - Adaptive Thresholding (Local dynamic neighborhood mean)
5. **Edge Detection & Morphology**:
   - Canny Edge Detection (Hysteresis thresholding)
   - Dilation & Erosion (Structuring element operations)
6. **Object Localization & Contours**:
   - `cv2.findContours` & hierarchical structure
   - Bounding boxes (`cv2.boundingRect`) & contour area calculation (`cv2.contourArea`)

## 🛠️ Setup & Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run scripts:
   ```bash
   # Process static image pipeline
   python image.py

   # Run live webcam feed
   python video.py
   ```
