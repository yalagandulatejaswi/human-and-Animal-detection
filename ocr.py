import easyocr
import cv2

class OCRPipeline:
    def __init__(self, languages=['en']):
        """Initialize OCR reader"""
        self.reader = easyocr.Reader(languages, gpu=True)
    
    def extract_text(self, image):
        """Extract text from image"""
        results = self.reader.readtext(image)
        
        texts = []
        for bbox, text, conf in results:
            texts.append({
                'text': text,
                'confidence': conf,
                'bbox': bbox
            })
        
        return texts
    
    def draw_text_boxes(self, image, ocr_results):
        """Draw OCR bounding boxes on image"""
        img = image.copy()
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            
            pts = [[int(x), int(y)] for x, y in bbox]
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
            cv2.putText(img, text, tuple(pts[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return img
