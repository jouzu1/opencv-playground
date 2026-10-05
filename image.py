import cv2
import numpy as np

# ==============================================================================
# 0. INPUT IMAGE
# ==============================================================================
# img = cv2.imread('img.jpg')       # Untuk sample gambar otsu threshold
# img = cv2.imread('simple.jpg')    # Untuk sample gambar simple threshold
img = cv2.imread('adaptive.jpg')    # Untuk sample gambar adaptive threshold

"""
==============================================================================
Steps to properly detect an object (Classical Computer Vision Pipeline):
1. Color Conversion         : Convert BGR to Grayscale (for shapes/contours) 
                              or HSV (for color detection)
2. Noise Reduction          : Blur the image (Gaussian/Median) to smooth out 
                              camera sensor noise
3. Binarization/Thresholding: Separate foreground from background using 
                              Thresholding (Simple/Otsu/Adaptive) OR Canny Edge Detection
4. Morphological Ops        : Clean up binary image (Erode/Opening to remove noise, 
                              Dilate/Closing to seal holes & connect lines)
5. Find Contours            : Extract vector coordinates of connected white pixels
6. Filter & Draw            : Discard small contours (e.g., if area > min_size) 
                              and draw bounding boxes / labels
==============================================================================
"""

# ==============================================================================
# 1. COLOR SPACES
# ==============================================================================
# Optional to implement HSV, Color spaces can be remarked back
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ==============================================================================
# 2. RESIZING & CROPPING
# ==============================================================================
# Optional to Resizing and cropping
cropped = img[100:400, 200:500]        # y1:y2, x1:x2
resized = cv2.resize(img, (500, 300))  # w x h

# ==============================================================================
# 3. DRAWING BOUNDING BOX / SHAPES
# ==============================================================================
# Optional to draw bounding box
# 1. Garis: cv2.line(img, pt1, pt2, color, thickness)
# cv2.line(img, (0,0), (200,200), (0,255,0), 2)

# 2. Lingkaran: cv2.circle(img, center, radius, color, thickness)
# cv2.circle(img, (200,200), 20, (0,255,0), 2)

# 3. Persegi panjang: cv2.rectangle(img, pt1, pt2, color, thickness)
# cv2.rectangle(img, (200,200), (400,400), (0,255,0), 2)

# 4. Elips: cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
# cv2.ellipse(img, (200,200), (400,400), 0, 0, 360, (0,255,0), 2)

# 5. Teks: cv2.putText(img, text, pt1, fontFace, fontScale, color, thickness)
# cv2.putText(img, 'Hello', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# ==============================================================================
# 4. BLURRING (SMOOTHING / NOISE REDUCTION)
# ==============================================================================
# Optional for Blur
# 1. Gaussian Blur: cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX)
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# 2. Median Blur: cv2.medianBlur(img, ksize)
median = cv2.medianBlur(img, 5)

# 3. Average Blur: cv2.blur(img, (kernel_size, kernel_size))
blur = cv2.blur(img, (7, 7))

# ==============================================================================
# 5. THRESHOLDING (BINARIZATION)
# ==============================================================================
# Optional for Thresholding

# 1. Simple Thresholding: cv2.threshold(src, thresh, maxval, type)
# Bahasa simplenya: "Kalo warnanya lebih terang dari thresh, jadiin putih, kalo lebih gelap, jadiin hitam"
ret, thresh = cv2.threshold(gray, 127.5, 255, cv2.THRESH_BINARY_INV)

# 2. Otsu Thresholding: cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Bahasa simplenya: "Gak usah nebak angka thresh manual, OpenCV otomatis nyari angka pemisah paling ideal dari histogram seluruh gambar"
ret2, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Adaptive Threshold: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# Bahasa simplenya: "Ngitung batas threshold dinamis di tiap area kecil (blok tetangga), jadi kebal bayangan dan pencahayaan yang belang-belang"
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# ==============================================================================
# 6. EDGE DETECTION & MORPHOLOGY
# ==============================================================================
# Optional for Canny Edge Detection: cv2.Canny(gray, Threshold1, Threshold2)
canny = cv2.Canny(gray, 100, 200)

# Optional for dilasi dan erosi: 
kernel = np.ones((5, 5), np.uint8)

# 1. Dilasi: cv2.dilate(gray, (5,5), iterations=1)
dilasi = cv2.dilate(gray, kernel, iterations=1)

# 2. Erosi: cv2.erode(gray, (5,5), iterations=1)
erosi = cv2.erode(gray, kernel, iterations=1)

# ==============================================================================
# 7. CONTOUR DETECTION
# ==============================================================================
# Deteksi kontur (menggunakan citra yang sudah di-blur dan thresholding)
# findContours mengembalikan list kontur dan struktur hierarki
# cv2.RETR_EXTERNAL hanya mengambil kontur luar saja
# cv2.CHAIN_APPROX_SIMPLE mengompresi segmen horizontal/vertikal menjadi endpoint saja
contours, hierarchy = cv2.findContours(
    adaptive,               # Bisa ganti thresh, tresh2, adaptive
    cv2.RETR_EXTERNAL,      # Bisa ganti RETR_EXTERNAL, RETR_LIST, RETR_TREE, RETR_CCOMP
    cv2.CHAIN_APPROX_SIMPLE # Bisa ganti CHAIN_APPROX_SIMPLE, CHAIN_APPROX_NONE
)

print(f"Ditemukan {len(contours)} buah kontur")
print(f"Hierarki {hierarchy}")

# ==============================================================================
# 8. CONTOUR FILTERING & BOUNDING BOX DRAWING
# ==============================================================================
# Buat salinan gambar untuk digambar kotaknya
img_with_box = img.copy()

# Loop setiap kontur yang ditemukan
for cnt in contours:
    # Hitung luas kontur
    area = cv2.contourArea(cnt)
    print(f"Area: {area}")

    if area > 50000: # Kondisi area berbeda beda tergantung dari gambar, tinggal di adjust dari variable area
        # Hitung bounding box (x, y, lebar, tinggi) dari kontur
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"x: {x}, y: {y}, w: {w}, h: {h}")
    
        # Gambar persegi panjang hijau dengan ketebalan 2
        # Parameter: gambar, titik sudut kiri atas, titik sudut kanan bawah, warna BGR, ketebalan
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # (Opsional) Tulis luas di atas kotak untuk debugging
        cv2.putText(
            img_with_box, 
            f"Area: {int(area)}", 
            (x + 10, y - 10 if y > 20 else y + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            1
        )

# ==============================================================================
# 9. DISPLAY & SAVE RESULT
# ==============================================================================
cv2.imshow('image', img_with_box) 
cv2.waitKey(0)           # Press 0 to close window
cv2.destroyAllWindows()  # closes the window after pressing any key

cv2.imwrite("Copy Image.jpg", img)