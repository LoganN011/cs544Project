import cv2
import numpy as np

class MoireDetector:
    def __init__(self, high_freq_radius_ratio=0.3, threshold=150.0):
        """
        Initializes the Moire Detector.
        :param high_freq_radius_ratio: The ratio of the image size to ignore as low frequencies.
        :param threshold: The threshold for the 99.5th percentile of high frequency magnitude. 
                          Scores above this are classified as screen/moire.
        """
        self.radius_ratio = high_freq_radius_ratio
        self.threshold = threshold

    def analyze(self, image):
        """
        Analyzes an image for Moiré patterns (screen spoofing).
        Returns (is_screen, score).
        """
        if image is None:
            return False, 0.0
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute 2D discrete Fourier Transform
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask to hide the central low-frequency region
        r = int(min(rows, cols) * self.radius_ratio)
        mask = np.ones((rows, cols), np.float32)
        
        # Create a circular mask
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
        mask[mask_area] = 0.0
        
        high_freq_mag = magnitude_spectrum * mask
        
        # Extract the high frequency region
        high_frequencies = high_freq_mag[mask > 0]
        if len(high_frequencies) == 0:
            return False, 0.0
            
        # Moire patterns from screen pixels appear as distinct bright spots
        # We calculate the 99.5th percentile to capture these robust peaks.
        score = np.percentile(high_frequencies, 99.5)
        
        is_screen = score > self.threshold
        return is_screen, score
