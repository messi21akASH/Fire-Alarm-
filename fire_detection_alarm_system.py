
import cv2
import torch
import numpy as np
import time
import threading
import pygame
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AudioAlarm:
    def __init__(self):
        self.alarm_active = False

    def continuous_alarm(self, stop_event):
        """Windows reliable alarm"""
        try:
            import winsound
            self.alarm_active = True

            while not stop_event.is_set():
                winsound.Beep(1200, 800)  # frequency, duration
                time.sleep(0.2)

            self.alarm_active = False
        except Exception as e:
            print("[ERROR] Alarm failed:", e)

class FireDetector:
    def __init__(self, model_type='yolov5'):
        self.model = None
        self.confidence_threshold = 0.6
        self.load_model(model_type)
        
    def load_model(self, model_type):
        """Load fire detection model"""
        try:
            if model_type == 'yolov5':
                # YOLOv5 for fire detection
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                print("[SUCCESS] YOLOv5 model loaded!")
            elif model_type == 'color':
                print("[INFO] Using color-based detection")
                self.model = None
        except Exception as e:
            print(f"[WARNING] Could not load ML model: {e}")
            print("[INFO] Using color-based detection as fallback")
            self.model = None
    
    def detect_fire(self, frame):
        """Detect fire in frame using ML or color-based method"""
        if self.model is not None:
            return self.detect_fire_ml(frame)
        else:
            return self.detect_fire_color(frame)
    
    def detect_fire_ml(self, frame):
        """Detect fire using YOLOv5"""
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        
        fire_detected = False
        max_confidence = 0
        
        for _, detection in detections.iterrows():
            if detection['name'] == 'fire' and detection['confidence'] > self.confidence_threshold:
                fire_detected = True
                max_confidence = max(max_confidence, detection['confidence'])
                
                # Draw bounding box
                x1, y1 = int(detection['xmin']), int(detection['ymin'])
                x2, y2 = int(detection['xmax']), int(detection['ymax'])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Fire: {detection['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame, fire_detected, max_confidence
    
    def detect_fire_color(self, frame):
        """Color-based fire detection (fallback)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red/orange color range for fire
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_detected = False
        fire_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                fire_detected = True
                fire_area += area
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        confidence = min(fire_area / (frame.shape[0] * frame.shape[1]), 1.0)
        
        if fire_detected:
            cv2.putText(frame, "FIRE DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame, fire_detected, confidence

class NotificationSystem:
    def __init__(self):
        self.email_enabled = False
        self.sms_enabled = False
        
    def send_email(self, location, confidence):
        """Send email alert"""
        if not self.email_enabled:
            return
            
        try:
            # Configure your email settings
            sender_email = "fire.alarm@system.com"
            receiver_email = "security@example.com"
            password = "your_password"
            
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = f"🔥 FIRE ALERT - {location}"
            
            body = f"""
            ⚠️ EMERGENCY ALERT ⚠️
            
            Fire detected at: {location}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Confidence: {confidence:.1%}
            
            Immediate action required!
            """
            
            message.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(message)
                
            print("[ALERT] Email sent!")
            
        except Exception as e:
            print(f"[ERROR] Email failed: {e}")
    
    def send_sms(self, message):
        """Send SMS alert (requires Twilio)"""
        if not self.sms_enabled:
            return
            
        try:
            from twilio.rest import Client
            
            # Configure your Twilio credentials
            account_sid = "your_account_sid"
            auth_token = "your_auth_token"
            
            client = Client(account_sid, auth_token)
            
            client.messages.create(
                body=message,
                from_="+1234567890",
                to="+0987654321"
            )
            
            print("[ALERT] SMS sent!")
            
        except ImportError:
            print("[INFO] Install Twilio: pip install twilio")
        except Exception as e:
            print(f"[ERROR] SMS failed: {e}")

class FireDetectionSystem:
    def __init__(self):
        self.detector = FireDetector()
        self.audio_alarm = AudioAlarm()
        self.notifier = NotificationSystem()
        
        # Detection parameters
        self.consecutive_detections = 0
        self.alarm_threshold = 3
        self.alarm_active = False
        self.stop_alarm_event = threading.Event()
        
        # Logging
        self.log_file = "fire_log.csv"
        self.setup_logging()
        
        print("=" * 50)
        print("🔥 FIRE DETECTION SYSTEM INITIALIZED 🔥")
        print("=" * 50)
    
    def setup_logging(self):
        """Setup log file"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,fire_detected,confidence,alarm_triggered\n")
    
    def log_event(self, detected, confidence, alarm):
        """Log event to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{detected},{confidence:.3f},{alarm}\n")
    
    def add_visual_alarm(self, frame):
        """Add visual alarm indicators to frame"""
        # Flash red border
        if int(time.time()) % 2 == 0:  # Blink every second
            cv2.rectangle(frame, (0, 0), 
                         (frame.shape[1], frame.shape[0]), 
                         (0, 0, 255), 10)
        
        # Warning text
        cv2.putText(frame, "🚨 FIRE ALARM 🚨", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return frame
    
    def trigger_alarms(self, confidence, location="Camera 1"):
        """Trigger all alarm systems"""
        print("\n" + "="*60)
        print("🚨🚨🚨 FIRE DETECTED! TRIGGERING ALARMS! 🚨🚨🚨")
        print("="*60)
        
        # Start audio alarm in separate thread
        self.stop_alarm_event.clear()
        audio_thread = threading.Thread(
            target=self.audio_alarm.continuous_alarm,
            args=(self.stop_alarm_event,)
        )
        audio_thread.daemon = True
        audio_thread.start()
        
        # Send notifications
        self.notifier.send_email(location, confidence)
        
        alert_msg = f"FIRE ALERT at {location}. Confidence: {confidence:.0%}. Time: {datetime.now()}"
        self.notifier.send_sms(alert_msg)
        
        # System notification (Linux/Mac)
        if os.name == 'posix':
            os.system(f'notify-send "FIRE DETECTED" "Confidence: {confidence:.0%}" -u critical -i dialog-warning')
    
    def stop_alarms(self):
        """Stop all alarms"""
        self.stop_alarm_event.set()
        self.alarm_active = False
        print("[INFO] Alarms stopped")
    
    def process_frame(self, frame):
        """Process single frame for fire detection"""
        # Detect fire
        processed_frame, detected, confidence = self.detector.detect_fire(frame)
        
        # Update detection counter
        if detected:
            self.consecutive_detections = min(self.consecutive_detections + 1, self.alarm_threshold + 1)
        else:
            self.consecutive_detections = max(self.consecutive_detections - 1, 0)
        
        # Check if alarm should be triggered
        alarm_triggered = False
        if self.consecutive_detections >= self.alarm_threshold:
            if not self.alarm_active:
                self.trigger_alarms(confidence)
                alarm_triggered = True
            self.alarm_active = True
            
            # Add visual alarm
            processed_frame = self.add_visual_alarm(processed_frame)
        else:
            if self.alarm_active:
                self.stop_alarms()
            self.alarm_active = False
        
        # Add status overlay
        status_color = (0, 0, 255) if detected else (0, 255, 0)
        status_text = "FIRE DETECTED!" if detected else "Normal"
        
        # Status bar
        cv2.rectangle(processed_frame, (0, 0), (frame.shape[1], 80), (30, 30, 30), -1)
        
        # Status text
        cv2.putText(processed_frame, f"Status: {status_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Confidence
        cv2.putText(processed_frame, f"Confidence: {confidence:.1%}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection counter
        counter_text = f"Detections: {self.consecutive_detections}/{self.alarm_threshold}"
        cv2.putText(processed_frame, counter_text, 
                   (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Alarm status
        alarm_text = "ALARM: ON" if self.alarm_active else "ALARM: OFF"
        alarm_color = (0, 0, 255) if self.alarm_active else (0, 255, 0)
        cv2.putText(processed_frame, alarm_text, 
                   (frame.shape[1] - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, alarm_color, 2)
        
        # Timestamp
        time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, time_text, 
                   (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Log event
        self.log_event(detected, confidence, alarm_triggered)
        
        return processed_frame, detected, confidence, alarm_triggered
    
    def run_webcam(self, camera_id=0):
        """Run fire detection on webcam feed"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return
        
        print("\n[INFO] Starting fire detection...")
        print("[CONTROLS]")
        print("  'q' - Quit")
        print("  's' - Silence alarm")
        print("  'r' - Reset detection counter")
        print("  'c' - Capture screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            # Process frame
            processed_frame, detected, confidence, alarm_triggered = self.process_frame(frame)
            
            # Display
            cv2.imshow('Fire Detection System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.stop_alarms()
                print("[INFO] Alarm silenced")
            elif key == ord('r'):
                self.consecutive_detections = 0
                print("[INFO] Detection counter reset")
            elif key == ord('c'):
                # Capture screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"[INFO] Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.stop_alarms()
        print("[INFO] System shutdown")
    
    def run_video_file(self, video_path):
        """Run detection on video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return
        
        print(f"[INFO] Processing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, _, _, _ = self.process_frame(frame)
            cv2.imshow('Fire Detection - Video', processed_frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def test_image(self, image_path):
        """Test detection on single image"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return
        
        processed_frame, detected, confidence, _ = self.process_frame(frame)
        
        print("\n" + "="*40)
        print("IMAGE ANALYSIS RESULT")
        print("="*40)
        print(f"Fire Detected: {'YES' if detected else 'NO'}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Detection Counter: {self.consecutive_detections}/{self.alarm_threshold}")
        print("="*40)
        
        cv2.imshow('Result', processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    print("🔥 FIRE DETECTION AND ALARM SYSTEM 🔥")
    print("Version 2.0 - Complete Solution")
    
    # Initialize system
    system = FireDetectionSystem()
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Live Webcam Detection")
        print("2. Process Video File")
        print("3. Test Single Image")
        print("4. View Log File")
        print("5. Exit")
        print("="*50)
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            camera_id = input("Enter camera ID (0 for default, 1 for external): ").strip()
            try:
                system.run_webcam(int(camera_id) if camera_id else 0)
            except ValueError:
                system.run_webcam(0)
                
        elif choice == '2':
            video_path = input("Enter video file path: ").strip()
            if os.path.exists(video_path):
                system.run_video_file(video_path)
            else:
                print("[ERROR] File not found")
                
        elif choice == '3':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                system.test_image(image_path)
            else:
                print("[ERROR] File not found")
                
        elif choice == '4':
            if os.path.exists(system.log_file):
                with open(system.log_file, 'r') as f:
                    print("\n" + "="*60)
                    print("FIRE DETECTION LOG")
                    print("="*60)
                    print(f.read())
            else:
                print("[INFO] No log file found")
                
        elif choice == '5':
            print("[INFO] Exiting system. Stay safe!")
            break
            
        else:
            print("[ERROR] Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    # Installation check
    try:
        import torch
        import cv2
        import pygame
        main()
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install torch torchvision opencv-python numpy pygame")
        print("\nFor YOLOv5 also run:")
        print("pip install pyyaml>=5.3.1")
