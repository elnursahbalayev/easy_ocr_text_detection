# Easy OCR Text Detection

## Overview
This project uses EasyOCR to detect and extract text from images. It draws bounding boxes around detected text and displays the results.

## Features
- Text detection in images using EasyOCR
- Confidence threshold filtering for detected text
- Visual display of detected text with bounding boxes

## Dependencies
- Python 3.x
- OpenCV (`cv2`)
- EasyOCR
- Matplotlib
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/easy_ocr_text_detection.git
cd easy_ocr_text_detection
```

2. Install the required dependencies:
```bash
pip install opencv-python easyocr matplotlib numpy
```

## Usage

1. Place your image in the project directory or use one of the sample images in the `data` directory.

2. Modify the `image_path` variable in `main.py` to point to your image:
```python
image_path = 'data/test1.png'  # Change this to your image path
```

3. Run the script:
```bash
python main.py
```

4. The script will:
   - Load the image
   - Detect text using EasyOCR
   - Draw bounding boxes around detected text with confidence above the threshold
   - Display the image with the detected text

## Example

```python
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to your image
image_path = 'data/test1.png'

# Read the image
img = cv2.imread(image_path)

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Detect text
text_ = reader.readtext(img)

# Set confidence threshold
threshold = 0.25

# Process and display detected text
for t_, t in enumerate(text_):
    print(t)
    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

# Display the image with detected text
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

## Sample Images
The `data` directory contains sample images you can use to test the text detection:
- `test1.png`
- `test2.png`
- `test3.png`

## Customization
- Adjust the `threshold` value to filter out low-confidence detections
- Change the language by modifying the EasyOCR reader initialization (e.g., `reader = easyocr.Reader(['en', 'fr'])` for English and French)
- Modify the visualization parameters in the `cv2.rectangle` and `cv2.putText` functions

## License
[Specify your license here]

## Acknowledgements
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for the OCR functionality
