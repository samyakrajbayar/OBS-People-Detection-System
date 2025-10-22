import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import time
import sys

# Configuration
OUTPUT_DIR = "detected_people"         
FPS = 10
FRAME_DELAY = int(1000 / FPS)  # milliseconds
CONFIDENCE_THRESHOLD = 0.5

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

def check_dependencies():
    """Check if all required packages are installed."""
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed. Run: pip install opencv-python")
        sys.exit(1)
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics installed")
    except ImportError:
        print("✗ Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    try:
        import numpy
        print(f"✓ NumPy installed")
    except ImportError:
        print("✗ NumPy not installed. Run: pip install numpy")
        sys.exit(1)

def load_model():
    """Load YOLOv8 model with error handling."""
    try:
        print("Loading YOLOv8 nano model...")
        model = YOLO('yolov8n.pt')
        print("✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("The model will be downloaded automatically on first run.")
        try:
            model = YOLO('yolov8n.pt')
            return model
        except Exception as e2:
            print(f"✗ Failed to load model: {e2}")
            sys.exit(1)

def find_video_source():
    """Find available video source (OBS Virtual Camera or webcam)."""
    print("\nSearching for video source...")
    
    # Try common OBS Virtual Camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"✓ Found video source at index {i}: {w}x{h}")
                return cap
            cap.release()
    
    print("✗ No video source found!")
    print("\nTroubleshooting:")
    print("1. For OBS: Enable Tools → Start Virtual Camera")
    print("2. Check if webcam is connected")
    print("3. Make sure no other app is using the camera")
    return None

def detect_people(frame, model):
    """
    Detect people in frame using YOLOv8.
    Returns (has_people, annotated_frame).
    """
    try:
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        has_people = False
        annotated_frame = frame.copy()
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is 'person'
                    has_people = True
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    conf = float(box.conf[0])
                    cv2.putText(annotated_frame, f'Person: {conf:.2f}', (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return has_people, annotated_frame
    except Exception as e:
        print(f"Error during detection: {e}")
        return False, frame

def save_frame(frame, frame_num):
    """Save frame with people detected."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"person_{timestamp}_{frame_num:06d}.jpg")
        success = cv2.imwrite(filename, frame)
        if success:
            print(f"✓ Saved: {filename}")
        else:
            print(f"✗ Failed to save: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving frame: {e}")
        return None

def main():
    print("=" * 50)
    print("OBS People Detection System")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Load model
    model = load_model()
    
    # Find video source
    cap = find_video_source()
    if cap is None:
        sys.exit(1)
    
    print(f"\nStarting detection at {FPS} FPS...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    people_detected_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Detect people
            has_people, annotated_frame = detect_people(frame, model)
            
            if has_people:
                people_detected_count += 1
                save_frame(frame, frame_count)
            
            # Display status on frame
            status = "PEOPLE DETECTED ✓" if has_people else "No people"
            color = (0, 255, 0) if has_people else (0, 0, 255)
            cv2.putText(annotated_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, color, 2)
            cv2.putText(annotated_frame, f"Frame: {frame_count} | People: {people_detected_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps_actual:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show preview
            cv2.imshow("OBS People Detection", annotated_frame)
            
            # Wait for next frame (10 FPS)
            key = cv2.waitKey(FRAME_DELAY) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        print(f"\n{'=' * 50}")
        print(f"Summary")
        print(f"{'=' * 50}")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with people: {people_detected_count}")
        print(f"Detection rate: {(people_detected_count/frame_count*100):.1f}%" if frame_count > 0 else "0%")
        print(f"Time elapsed: {elapsed_time:.1f}s")
        print(f"Actual FPS: {frame_count/elapsed_time:.1f}")
        print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"{'=' * 50}")

if __name__ == "__main__":
    main()
