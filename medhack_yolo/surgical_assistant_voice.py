"""
Voice-Enabled YOLOv11 Surgical Assistance System
Real-time organ detection with voice alerts and voice commands
"""

import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time
from collections import deque
from gtts import gTTS
import speech_recognition as sr
import os
import tempfile
import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'best.pt'
VIDEO_SOURCE = 'iv2.mp4'  # 0 for webcam, or path to video file
CONF_THRESHOLD = 0.40

# Voice Alert Settings
ENABLE_VOICE_ALERTS = True
ALERT_ON_FIRST_DETECTION = False
ALERT_ON_DISAPPEARANCE = False
REANNOUNCE_AFTER_SECONDS = 20  # Re-announce if missing for X seconds

# Voice Control Settings
ENABLE_VOICE_CONTROL = True
VOICE_COMMAND_TIMEOUT = 2  # seconds

# Display Settings
SHOW_MASKS = True
SHOW_LABELS = True
SHOW_BOXES = False
MASK_ALPHA = 0.4

# Organ names mapping (clean names for voice)
ORGAN_NAMES = {
    0: "External Iliac Artery",
    1: "External Iliac Vein",
    2: "Obturator Nerve",
    3: "Ovary",
    4: "Ureter",
    5: "Uterine Artery",
    6: "Uterus"
}

# ============================================================================
# TEXT-TO-SPEECH ENGINE
# ============================================================================

class VoiceAnnouncer:
    """Thread-safe voice announcement system using Google TTS."""
    
    def __init__(self):
        print("🔊 TTS engine ready (Google TTS)")
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        """Background worker for speech synthesis."""
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if text:
                    print(f"🔊 Voice: {text}")
                    
                    # Generate speech with Google TTS
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                        temp_file = f.name
                    
                    try:
                        # Create TTS
                        tts = gTTS(text=text, lang='en', slow=False)
                        tts.save(temp_file)
                        
                        # Play audio using system player
                        if os.name == 'posix':  # Linux/Mac
                            # Try mpg123, then mpg321, then ffplay
                            for player in ['mpg123', 'mpg321', 'ffplay -nodisp -autoexit']:
                                try:
                                    subprocess.run(player.split() + [temp_file], 
                                                 stdout=subprocess.DEVNULL, 
                                                 stderr=subprocess.DEVNULL,
                                                 check=True)
                                    break
                                except (subprocess.CalledProcessError, FileNotFoundError):
                                    continue
                        elif os.name == 'nt':  # Windows
                            os.system(f'start /min "" "{temp_file}"')
                            time.sleep(2)  # Wait for playback
                    finally:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except:
                            pass
                    
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice error: {e}")
                continue
            except Exception as e:
                print(f"Voice error: {e}")
    
    def announce(self, text):
        """Queue text for announcement."""
        if not self.queue.full():
            self.queue.put(text)
    
    def stop(self):
        """Stop the voice announcer."""
        self.running = False
        self.thread.join()

# ============================================================================
# VOICE COMMAND LISTENER
# ============================================================================

class VoiceCommandListener:
    """Listen for voice commands using speech recognition."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()
        self.running = True
        self.listening = False
        
        # Adjust for ambient noise
        print("🎤 Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        self.thread = threading.Thread(target=self._listen_worker, daemon=True)
        self.thread.start()
        print("✅ Voice command system ready")
    
    def _listen_worker(self):
        """Background worker for voice recognition."""
        while self.running:
            if not self.listening:
                time.sleep(0.1)
                continue
            
            try:
                with self.microphone as source:
                    # Listen with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                # Recognize speech
                text = self.recognizer.recognize_google(audio).lower()
                print(f"🎤 Heard: {text}")
                self.command_queue.put(text)
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                if self.running:
                    print(f"Voice recognition error: {e}")
    
    def start_listening(self):
        """Enable voice command listening."""
        self.listening = True
        print("🎤 Listening for commands...")
    
    def stop_listening(self):
        """Disable voice command listening."""
        self.listening = False
    
    def get_command(self):
        """Get next command from queue (non-blocking)."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the listener."""
        self.running = False
        self.thread.join()

# ============================================================================
# DETECTION TRACKER
# ============================================================================

class DetectionTracker:
    """Track organ detections for voice alerts."""
    
    def __init__(self, reannounce_after=10):
        self.current_detections = set()
        self.last_seen = {}  # {organ_id: timestamp}
        self.last_announced = {}  # {organ_id: timestamp}
        self.reannounce_after = reannounce_after
    
    def update(self, detected_organs):
        """Update detections and return events."""
        current_time = time.time()
        detected_set = set(detected_organs)
        
        events = {
            'new': [],
            'disappeared': [],
            'reappeared': []
        }
        
        # Check for new detections
        for organ_id in detected_set:
            if organ_id not in self.current_detections:
                # New detection
                if organ_id not in self.last_announced:
                    events['new'].append(organ_id)
                    self.last_announced[organ_id] = current_time
                else:
                    # Check if it's been gone long enough to re-announce
                    time_since_last = current_time - self.last_announced[organ_id]
                    if time_since_last > self.reannounce_after:
                        events['reappeared'].append(organ_id)
                        self.last_announced[organ_id] = current_time
            
            self.last_seen[organ_id] = current_time
        
        # Check for disappeared detections
        for organ_id in self.current_detections:
            if organ_id not in detected_set:
                events['disappeared'].append(organ_id)
        
        self.current_detections = detected_set
        return events

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class SurgicalAssistanceSystem:
    """Main application with voice features."""
    
    def __init__(self):
        print("=" * 70)
        print("🏥 Voice-Enabled Surgical Assistance System")
        print("=" * 70)
        
        # Load model
        print(f"\n[1/4] Loading model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        print(f"      ✅ Model loaded")
        print(f"      Classes: {list(ORGAN_NAMES.values())}")
        
        # Initialize voice systems
        print(f"\n[2/4] Initializing voice systems...")
        self.voice_announcer = VoiceAnnouncer() if ENABLE_VOICE_ALERTS else None
        self.voice_listener = VoiceCommandListener() if ENABLE_VOICE_CONTROL else None
        
        if ENABLE_VOICE_CONTROL:
            self.voice_listener.start_listening()
        
        # Initialize tracker
        self.tracker = DetectionTracker(reannounce_after=REANNOUNCE_AFTER_SECONDS)
        
        # State
        self.show_masks = SHOW_MASKS
        self.show_labels = SHOW_LABELS
        self.show_boxes = SHOW_BOXES
        self.paused = False
        self.filter_organs = None  # None = show all, or set of organ IDs
        
        # Open video
        print(f"\n[3/4] Opening video source: {VIDEO_SOURCE}")
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")
        
        # Get properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"      Resolution: {self.width}x{self.height} @ {self.fps} FPS")
        
        print(f"\n[4/4] System ready!")
        print("=" * 70)
        print("\n📋 VOICE COMMANDS:")
        print("  • 'hide overlays' or 'hide masks'")
        print("  • 'show overlays' or 'show masks'")
        print("  • 'show [organ name]' - e.g., 'show ureter'")
        print("  • 'show vessels' - show arteries and veins")
        print("  • 'show all' - show all structures")
        print("  • 'pause' or 'pause AI'")
        print("  • 'resume' or 'resume AI'")
        print("  • 'what structures' or 'what visible' - query current view")
        print("\n⌨️  KEYBOARD SHORTCUTS:")
        print("  • 'q' - Quit")
        print("  • 'm' - Toggle masks")
        print("  • 'l' - Toggle labels")
        print("  • 'p' - Pause/Resume")
        print("=" * 70 + "\n")
    
    def process_voice_command(self, command):
        """Process voice commands."""
        if not command:
            return
        
        # Hide overlays
        if 'hide' in command and ('overlay' in command or 'mask' in command):
            self.show_masks = False
            self.show_labels = False
            if self.voice_announcer:
                self.voice_announcer.announce("Overlays hidden")
        
        # Show overlays
        elif 'show' in command and ('overlay' in command or 'mask' in command):
            self.show_masks = True
            self.show_labels = True
            self.filter_organs = None
            if self.voice_announcer:
                self.voice_announcer.announce("Overlays shown")
        
        # Show specific organ
        elif 'show' in command:
            if 'ureter' in command:
                self.filter_organs = {4}  # Ureter
                self.show_masks = True
                if self.voice_announcer:
                    self.voice_announcer.announce("Showing ureter only")
            
            elif 'nerve' in command or 'obturator' in command:
                self.filter_organs = {2}  # Obturator nerve
                self.show_masks = True
                if self.voice_announcer:
                    self.voice_announcer.announce("Showing nerve only")
            
            elif 'vessel' in command or 'artery' in command or 'vein' in command:
                self.filter_organs = {0, 1, 5}  # Arteries and veins
                self.show_masks = True
                if self.voice_announcer:
                    self.voice_announcer.announce("Showing vessels only")
            
            elif 'all' in command:
                self.filter_organs = None
                self.show_masks = True
                if self.voice_announcer:
                    self.voice_announcer.announce("Showing all structures")
        
        # Pause/Resume
        elif 'pause' in command:
            self.paused = True
            if self.voice_announcer:
                self.voice_announcer.announce("AI paused")
        
        elif 'resume' in command or 'continue' in command:
            self.paused = False
            if self.voice_announcer:
                self.voice_announcer.announce("AI resumed")
        
        # Query visible structures
        elif 'what' in command and ('structure' in command or 'visible' in command):
            if self.tracker.current_detections:
                organs = [ORGAN_NAMES[oid] for oid in self.tracker.current_detections]
                if len(organs) == 1:
                    response = f"{organs[0]} is visible"
                elif len(organs) == 2:
                    response = f"{organs[0]} and {organs[1]} are visible"
                else:
                    response = f"{', '.join(organs[:-1])}, and {organs[-1]} are visible"
            else:
                response = "No structures currently visible"
            
            if self.voice_announcer:
                self.voice_announcer.announce(response)
    
    def announce_events(self, events):
        """Announce detection events."""
        if not self.voice_announcer or not ENABLE_VOICE_ALERTS:
            return
        
        # Announce new detections
        if ALERT_ON_FIRST_DETECTION:
            for organ_id in events['new']:
                self.voice_announcer.announce(f"{ORGAN_NAMES[organ_id]} identified")
        
        # Announce reappearances (disabled - too noisy)
        # for organ_id in events['reappeared']:
        #     self.voice_announcer.announce(f"{ORGAN_NAMES[organ_id]} back in view")
        
        # Announce disappearances
        if ALERT_ON_DISAPPEARANCE:
            for organ_id in events['disappeared']:
                self.voice_announcer.announce(f"{ORGAN_NAMES[organ_id]} no longer visible")
    
    def draw_frame(self, frame, results):
        """Draw detection results on frame."""
        if results[0].masks is None or not self.show_masks:
            return frame
        
        # Use YOLO's default plot for consistent colors with transparency
        annotated_frame = results[0].plot(
            boxes=False,
            labels=False,
            conf=False,
            masks=self.show_masks
        )
        
        # Blend with original frame for transparency
        annotated_frame = cv2.addWeighted(frame, 1 - MASK_ALPHA, annotated_frame, MASK_ALPHA, 0)
        
        # Add custom labels if enabled
        if self.show_labels and results[0].masks is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
            for box, cls_id in zip(boxes, classes):
                # Apply organ filter
                if self.filter_organs is not None and cls_id not in self.filter_organs:
                    continue
                
                # Get label position
                x1, y1, x2, y2 = map(int, box)
                label_x = (x1 + x2) // 2
                label_y = (y1 + y2) // 2
                
                # Get label text
                label = ORGAN_NAMES.get(int(cls_id), f"Class {cls_id}")
                
                # Draw label with background
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(label, font, 0.5, 1)
                cv2.rectangle(annotated_frame, (label_x - 5, label_y - text_h - 5),
                            (label_x + text_w + 5, label_y + 5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, label, (label_x, label_y),
                          font, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def run(self):
        """Main loop."""
        fps_history = deque(maxlen=30)
        
        try:
            while True:
                frame_start = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process voice commands
                if self.voice_listener:
                    command = self.voice_listener.get_command()
                    if command:
                        self.process_voice_command(command)
                
                # Run inference (if not paused)
                if not self.paused:
                    results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
                    
                    # Track detections
                    detected_organs = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                    events = self.tracker.update(detected_organs)
                    self.announce_events(events)
                    
                    # Draw
                    frame = self.draw_frame(frame, results)
                else:
                    # Show paused indicator
                    cv2.putText(frame, "⏸ PAUSED", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                fps_history.append(fps)
                avg_fps = np.mean(fps_history)
                
                # Display info
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show active organs
                if self.tracker.current_detections:
                    y_pos = 100
                    for organ_id in sorted(self.tracker.current_detections):
                        cv2.putText(frame, f"[*] {ORGAN_NAMES[organ_id]}", (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_pos += 25
                
                # Display
                cv2.imshow('Surgical Assistance System', frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.show_masks = not self.show_masks
                elif key == ord('l'):
                    self.show_labels = not self.show_labels
                elif key == ord('p'):
                    self.paused = not self.paused
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        print("\n🧹 Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self.voice_announcer:
            self.voice_announcer.stop()
        
        if self.voice_listener:
            self.voice_listener.stop()
        
        print("✅ Shutdown complete")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        system = SurgicalAssistanceSystem()
        system.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
