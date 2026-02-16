import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from detection import ObjectDetector
from classification import Classifier
from ocr import OCRPipeline
from utils import ensure_dirs, create_video_from_frames

# Initialize
ensure_dirs()

st.set_page_config(page_title="Human & Animal Detection", layout="wide")
st.title("🎥 Human & Animal Detection System")

# Sidebar
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.1, 0.05)
st.sidebar.caption("Lower = more detections (may include false positives)")
enable_ocr = st.sidebar.checkbox("Enable OCR", value=False)

st.sidebar.markdown("---")
st.sidebar.info("🔍 Yellow boxes = Unknown objects (for debugging)")
st.sidebar.info("Check your terminal/console for debug messages")

# Initialize models
@st.cache_resource
def load_models():
    detector = ObjectDetector(model_path=None)
    classifier = Classifier()
    ocr = OCRPipeline() if enable_ocr else None
    return detector, classifier, ocr

detector, classifier, ocr = load_models()

def process_frame_with_count(frame, detector, classifier, conf_threshold, enable_ocr, ocr):
    """Process a single frame/image and return detection count"""
    # Detect all objects
    results = detector.detect(frame, conf_threshold)
    crops = detector.crop_detections(frame, results)
    
    detection_count = 0
    
    print(f"DEBUG: Found {len(crops)} detections")
    
    # Classify each detected object
    for idx, crop_data in enumerate(crops):
        crop_img = crop_data['image']
        x1, y1, x2, y2 = crop_data['box']
        det_conf = crop_data['conf']
        class_name = crop_data.get('class_name', '')
        
        print(f"DEBUG: Detection {idx}: COCO class='{class_name}', conf={det_conf:.2f}")
        
        if crop_img.size == 0:
            continue
        
        # First check COCO class name from detector
        if class_name == 'person':
            category = 'human'
            print(f"DEBUG: Detected as HUMAN (from COCO)")
        elif class_name in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                           'elephant', 'bear', 'zebra', 'giraffe']:
            category = 'animal'
            print(f"DEBUG: Detected as ANIMAL (from COCO)")
        else:
            # Use classifier for unknown objects
            category, class_conf = classifier.classify(crop_img)
            print(f"DEBUG: Classified as {category} (from ImageNet)")
        
        # Don't skip unknown - show them for debugging
        if category == 'unknown':
            # Still draw it but in yellow for debugging
            color = (0, 255, 255)  # YELLOW for unknown
            label_text = f"Unknown ({class_name})"
        elif category == 'human':
            color = (0, 255, 0)  # GREEN
            label_text = "Human"
            detection_count += 1
        else:  # animal
            color = (0, 0, 255)  # RED
            label_text = "Animal"
            detection_count += 1
        
        # Draw square bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Add label with background INSIDE the box
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Position text inside box at top
        text_x = x1 + 5
        text_y = y1 + text_height + 10
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + 5), color, -1)
        
        # Draw text in white
        cv2.putText(frame, label_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add confidence score at bottom of box
        conf_text = f"{det_conf:.2f}"
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        conf_x = x1 + 5
        conf_y = y2 - 10
        
        cv2.rectangle(frame, (conf_x - 3, conf_y - conf_h - 3), 
                     (conf_x + conf_w + 3, conf_y + 3), color, -1)
        cv2.putText(frame, conf_text, (conf_x, conf_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # OCR if enabled
    if enable_ocr and ocr:
        ocr_results = ocr.extract_text(frame)
        frame = ocr.draw_text_boxes(frame, ocr_results)
    
    return frame, detection_count

def process_frame(frame, detector, classifier, conf_threshold, enable_ocr, ocr):
    """Process a single frame/image"""
    processed, _ = process_frame_with_count(frame, detector, classifier, conf_threshold, enable_ocr, ocr)
    return processed

# Upload file
uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])

if uploaded_file:
    file_type = uploaded_file.type
    
    # Handle IMAGE
    if file_type.startswith('image'):
        # Display original
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        if st.button("Process Image"):
            with st.spinner("Processing..."):
                # Process
                processed = process_frame(img_bgr, detector, classifier, conf_threshold, enable_ocr, ocr)
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                
                # Display result
                with col2:
                    st.subheader("Detected Objects")
                    st.image(processed_rgb, use_container_width=True)
                
                # Save and download
                output_path = 'outputs/processed_image.jpg'
                cv2.imwrite(output_path, processed)
                
                with open(output_path, 'rb') as f:
                    st.download_button("Download Result", f, file_name="detected.jpg", mime="image/jpeg")
                
                st.success("✅ Detection complete!")
    
    # Handle VIDEO
    else:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("Process Video"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_placeholder = st.empty()
            
            status_text.info("🔄 Opening video file...")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                st.error("❌ Failed to open video file!")
                st.stop()
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            status_text.info(f"📹 Video: {total_frames} frames, {fps} FPS, {width}x{height}")
            
            processed_frames = []
            frame_count = 0
            detection_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed, num_detections = process_frame_with_count(frame, detector, classifier, conf_threshold, enable_ocr, ocr)
                processed_frames.append(processed)
                detection_count += num_detections
                
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                
                # Update status every 10 frames
                if frame_count % 10 == 0:
                    status_text.text(f"Processing frame {frame_count}/{total_frames} | Detections so far: {detection_count}")
            
            cap.release()
            
            status_text.success(f"✅ Processed {frame_count} frames with {detection_count} total detections!")
            
            if detection_count == 0:
                st.warning("⚠️ No humans or animals detected. Try lowering the confidence threshold or use a different video.")
            
            # Save output video
            output_path = 'outputs/processed_video.mp4'
            status_text.info("💾 Saving output video...")
            create_video_from_frames(processed_frames, output_path, fps)
            
            st.success("✅ Processing complete!")
            st.video(output_path)
            
            # Download button
            with open(output_path, 'rb') as f:
                st.download_button("Download Processed Video", f, file_name="detected.mp4")

# Legend
st.sidebar.markdown("---")
st.sidebar.markdown("### Color Legend")
st.sidebar.markdown("🟢 **Green Box** = Human")
st.sidebar.markdown("🔴 **Red Box** = Animal")
