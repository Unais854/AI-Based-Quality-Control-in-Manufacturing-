# ================================
# Imports
# ================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ================================
# 1. Create Test Images (Synthetic)
# ================================
def create_test_images():
    """Create example test images for QC simulation."""
    if not os.path.exists("qc_test_images"):
        os.makedirs("qc_test_images")

    # Reference (Good) Image
    reference = np.ones((300, 400), dtype=np.uint8) * 200
    cv2.rectangle(reference, (100, 100), (300, 200), 150, -1)
    cv2.circle(reference, (200, 150), 40, 180, -1)
    cv2.imwrite("qc_test_images/reference_good.jpg", reference)

    # Defective Image: Scratch
    test1 = reference.copy()
    cv2.line(test1, (150, 120), (250, 180), 100, 3)
    cv2.imwrite("qc_test_images/test_with_scratch.jpg", test1)

    # Defective Image: Spot
    test2 = reference.copy()
    cv2.circle(test2, (280, 130), 15, 100, -1)
    cv2.imwrite("qc_test_images/test_with_spot.jpg", test2)

    # Another Good Image
    cv2.imwrite("qc_test_images/test_good.jpg", reference)

    print("Test images created in 'qc_test_images' directory.")
    return ["reference_good.jpg", "test_with_scratch.jpg", "test_with_spot.jpg", "test_good.jpg"]

# ================================
# 2. QC Vision Class
# ================================
class SimpleQCVision:
    def __init__(self, reference_image_path=None):
        """Initialize the QC Vision system with optional reference."""
        if reference_image_path and os.path.exists(reference_image_path):
            self.reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
            print(f"Loaded reference image: {reference_image_path}")
        else:
            self.reference_image = None
            print("No reference image provided. Using unsupervised mode.")

        self.threshold = 30
        self.min_defect_size = 50

    def detect_defects(self, image_path):
        """Detect defects in the test image."""
        test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if test_image is None:
            print(f"Error loading image: {image_path}")
            return None, []

        vis_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
        test_image_blur = cv2.GaussianBlur(test_image, (5, 5), 0)

        if self.reference_image is not None:
            defects = self._detect_with_reference(test_image_blur, vis_image)
        else:
            defects = self._detect_without_reference(test_image_blur, vis_image)

        return vis_image, defects

    def _detect_with_reference(self, test_image, vis_image):
        """Detect differences using reference comparison."""
        ref = cv2.resize(self.reference_image, (test_image.shape[1], test_image.shape[0]))
        ref = cv2.GaussianBlur(ref, (5, 5), 0)
        diff = cv2.absdiff(ref, test_image)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return self._extract_defects(contours, test_image, vis_image)

    def _detect_without_reference(self, test_image, vis_image):
        """Unsupervised defect detection using thresholding."""
        thresh = cv2.adaptiveThreshold(test_image, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return self._extract_defects(contours, test_image, vis_image)

    def _extract_defects(self, contours, test_image, vis_image):
        """Extract defect data and draw annotations."""
        defects = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area >= self.min_defect_size:
                x, y, w, h = cv2.boundingRect(cnt)
                defect_type = self._classify_defect(test_image, x, y, w, h, area)
                defects.append({
                    "id": i + 1,
                    "type": defect_type,
                    "area": area,
                    "position": (x + w // 2, y + h // 2)
                })
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(vis_image, defect_type, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(vis_image, f"Defects: {len(defects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return defects

    def _classify_defect(self, image, x, y, w, h, area):
        """Classify defect based on shape and intensity."""
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            return "Unknown"

        aspect_ratio = w / h if h != 0 else 0
        mean_intensity = np.mean(roi)

        if aspect_ratio > 3:
            return "Scratch"
        elif area < 200:
            return "Spot"
        elif mean_intensity < 100:
            return "Dark Area"
        elif mean_intensity > 200:
            return "Light Area"
        else:
            return "Surface Defect"

# ================================
# 3. Save Annotated Results
# ================================
def save_results(vis_image, defects, original_image_name):
    """Save annotated image and defect report."""
    if not os.path.exists("qc_results"):
        os.makedirs("qc_results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(original_image_name)[0]
    image_output = f"qc_results/{base_name}_{timestamp}.jpg"
    text_output = f"qc_results/{base_name}_{timestamp}_defects.txt"

    # Save annotated image
    cv2.imwrite(image_output, vis_image)

    # Save defect report
    with open(text_output, 'w') as f:
        f.write(f"Defect Analysis for {original_image_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total defects: {len(defects)}\n\n")
        for defect in defects:
            f.write(f"Defect #{defect['id']}\n")
            f.write(f"  Type: {defect['type']}\n")
            f.write(f"  Area: {defect['area']:.2f} pixels\n")
            f.write(f"  Position: {defect['position']}\n\n")

    return image_output

# ================================
# 4. Run Complete QC Pipeline
# ================================
def run_qc():
    """Run the quality control inspection process."""
    print("Launching Quality Control System...\n" + "=" * 50)

    # Create or load images
    if not os.path.exists("qc_test_images") or not os.listdir("qc_test_images"):
        image_files = create_test_images()
    else:
        image_files = [f for f in os.listdir("qc_test_images") if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        print("No test images found.")
        return

    reference_path = os.path.join("qc_test_images", image_files[0])
    qc_system = SimpleQCVision(reference_path)

    # Analyze each image
    for image_file in image_files[1:]:
        print(f"\nProcessing {image_file}...")
        image_path = os.path.join("qc_test_images", image_file)
        vis_image, defects = qc_system.detect_defects(image_path)

        if vis_image is not None:
            print(f"Detected {len(defects)} defect(s):")
            for defect in defects:
                print(f"  - ID #{defect['id']}: {defect['type']} | Area: {defect['area']:.2f}")

            output_path = save_results(vis_image, defects, image_file)
            print(f"Results saved: {output_path}")

            # Display results
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Defect Analysis - {image_file}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    print("\nQC inspection finished.")

# ================================
# 5. Main Entry Point
# ================================
if __name__ == "__main__":
    run_qc()
