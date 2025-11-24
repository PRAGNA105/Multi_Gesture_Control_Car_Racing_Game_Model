# main_voice_commentary.py - F1 3D Gesture-Driven Racing Game
# FIXED: Emotion Detection + Hand Gestures + Blink Controls + Enhanced GUI

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import random
import math
from collections import deque
import pyttsx3
from threading import Thread
import os

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Screen dimensions
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🏎️ F1 Gesture Racer Pro - Enhanced Edition")

# Colors - Enhanced palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 150, 255)
YELLOW = (255, 220, 0)
ORANGE = (255, 140, 0)
PURPLE = (200, 50, 255)
CYAN = (0, 255, 255)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (180, 180, 180)
SKY_BLUE = (135, 206, 250)
GRASS_GREEN = (34, 139, 34)
NEON_GREEN = (0, 255, 100)
NEON_PINK = (255, 16, 240)
NEON_BLUE = (0, 255, 255)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,  # Increased for better accuracy
    min_tracking_confidence=0.7
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,  # Increased for better accuracy
    min_tracking_confidence=0.7
)

# Text-to-Speech - Fixed initialization
engine = None
engine_lock = Thread()

def init_tts():
    """Initialize TTS engine safely"""
    global engine
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 0.9)
        return True
    except Exception as e:
        print(f"⚠️ TTS initialization failed: {e}")
        return False

def speak(text):
    """Non-blocking TTS with error handling"""
    def _speak():
        global engine
        try:
            if engine is None:
                if not init_tts():
                    print(f"🔊 {text}")
                    return
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"🔊 {text}")
    Thread(target=_speak, daemon=True).start()

# Initialize TTS at startup
init_tts()


class ParticleSystem:
    """Advanced particle effects system"""
    
    def __init__(self):
        self.particles = []
    
    def emit(self, x, y, ptype='spark', count=5, velocity_scale=1):
        """Emit particles of different types"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8) * velocity_scale
            
            particle = {
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'type': ptype,
                'life': 1.0,
                'size': random.uniform(3, 8),
                'color': self._get_particle_color(ptype)
            }
            self.particles.append(particle)
    
    def _get_particle_color(self, ptype):
        colors = {
            'spark': NEON_GREEN,
            'boost': NEON_PINK,
            'explosion': ORANGE,
            'hit': RED,
            'smoke': LIGHT_GRAY
        }
        return colors.get(ptype, WHITE)
    
    def update(self):
        """Update all particles"""
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.15
            p['life'] -= 0.02
            
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def draw(self, surface):
        """Draw all particles"""
        for p in self.particles:
            alpha = int(255 * p['life'])
            color = p['color']
            size = int(p['size'] * p['life'])
            
            if size > 0:
                pygame.draw.circle(surface, color, (int(p['x']), int(p['y'])), size)


class GlowEffect:
    """Neon glow effects"""
    
    @staticmethod
    def draw_neon_circle(surface, pos, radius, color, glow_size=20):
        """Draw a neon glowing circle"""
        glow_surf = pygame.Surface((radius * 2 + glow_size * 2, radius * 2 + glow_size * 2), pygame.SRCALPHA)
        
        for i in range(glow_size, 0, -1):
            alpha = int(100 * (1 - i / glow_size))
            glow_color = (*color, alpha)
            pygame.draw.circle(glow_surf, glow_color, (radius + glow_size, radius + glow_size), radius + i)
        
        pygame.draw.circle(glow_surf, (*color, 255), (radius + glow_size, radius + glow_size), radius)
        surface.blit(glow_surf, (pos[0] - radius - glow_size, pos[1] - radius - glow_size))

class EmotionDetector:
    """FIXED & ENHANCED: More stable and accurate emotion detection"""

    def __init__(self):
        self.last_emotion = {"emotion": "neutral", "confidence": 0.0}
        self.last_detection_time = 0
        self.detection_interval = 1.2  # Detect every ~1.2s for smoother results
        self.emotion_history = deque(maxlen=7)
        self.deepface_available = False
        self.confidence_threshold = 20  # Adjusted threshold for real-world faces

        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.deepface_available = True
            print("✅ DeepFace loaded successfully!")
        except ImportError:
            print("⚠️ DeepFace not available. Install with: pip install deepface")
            self.DeepFace = None

        # Load OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _preprocess_face(self, frame, x, y, w, h):
        """Preprocess face ROI for DeepFace"""
        # Add generous padding
        pad = int(h * 0.25)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        face_roi = frame[y1:y2, x1:x2]

        # Resize and normalize lighting
        face_resized = cv2.resize(face_roi, (224, 224))
        face_resized = cv2.convertScaleAbs(face_resized, alpha=1.15, beta=15)
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        face_resized = cv2.merge((l, a, b))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_LAB2BGR)
        return face_resized

    def detect_emotion(self, frame):
        """Stable emotion detection with history smoothing"""
        if not self.deepface_available:
            return self.last_emotion

        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.last_emotion

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))

            if len(faces) == 0:
                # Keep last emotion
                return self.last_emotion

            # Choose the largest face (most likely main user)
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_resized = self._preprocess_face(frame, x, y, w, h)

            # DeepFace analysis
            result = self.DeepFace.analyze(
                face_resized,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True
            )

            # Handle DeepFace returning list or dict
            if isinstance(result, list):
                result = result[0]

            emotion = result.get("dominant_emotion", "neutral")
            emotion_scores = result.get("emotion", {})
            confidence = float(emotion_scores.get(emotion, 0))

            # Normalize confidence (if in percentage)
            if confidence > 1.0:
                confidence = min(confidence, 100.0)

            # Ignore low-confidence detections
            if confidence < self.confidence_threshold:
                return self.last_emotion

            # Add to history
            self.emotion_history.append(emotion)
            self.last_detection_time = current_time

            # Weighted majority voting (newest counts more)
            weights = {e: (i + 1) for i, e in enumerate(self.emotion_history)}
            most_common = max(set(self.emotion_history), key=lambda e: sum(w for emo, w in weights.items() if emo == e))

            self.last_emotion = {"emotion": most_common, "confidence": confidence}
            return self.last_emotion

        except Exception as e:
            print(f"⚠️ Emotion detection error: {e}")
            return self.last_emotion


        
class GestureDetector:
    """FIXED: Improved gesture and blink detection"""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # IMPROVED BLINK DETECTION PARAMETERS
        self.blink_times = deque(maxlen=30)
        self.eye_closed_frames = 0
        self.eyes_were_open = True
        self.blink_threshold = 0.20  # Optimized threshold
        self.blink_count = 0
        self.last_blink_time = 0
        self.double_blink_window = 0.7  # 700ms window
        
        self.gesture_history = deque(maxlen=8)  # Shorter for faster response
        self.current_gesture = None
        
        self.emotion_detector = EmotionDetector()
        self.last_emotion_check = 0
        
        self.hand_position_history = deque(maxlen=3)  # Shorter for faster response
        
        # EAR history for smoothing
        self.ear_history = deque(maxlen=3)
        
        # Gesture confidence tracking
        self.last_confident_gesture = None
        self.gesture_confidence_counter = 0
    
    def calculate_ear(self, landmarks, eye_indices):
        """IMPROVED EAR calculation"""
        try:
            def dist(p1, p2):
                return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
            
            # Vertical distances
            v1 = dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
            v2 = dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
            
            # Horizontal distance
            h = dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
            
            if h > 0:
                ear = (v1 + v2) / (2.0 * h)
                return ear
            return 0.25
            
        except (IndexError, AttributeError, ZeroDivisionError):
            return 0.25
    
    def detect_blink_type(self, face_landmarks):
        """Reliable blink detection for short, long, and double blink"""
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        try:
            left_ear = self.calculate_ear(face_landmarks.landmark, LEFT_EYE)
            right_ear = self.calculate_ear(face_landmarks.landmark, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # Smooth EAR
            self.ear_history.append(avg_ear)
            smoothed_ear = sum(self.ear_history) / len(self.ear_history)

            current_time = time.time()
            blink_type = None

            # Eye closed
            if smoothed_ear < self.blink_threshold:
                self.eye_closed_frames += 1
                if self.eyes_were_open:
                    self.eyes_were_open = False

            else:
                # Eye just opened → blink occurred
                if not self.eyes_were_open:
                    blink_duration = self.eye_closed_frames / 30.0  # seconds
                    self.eye_closed_frames = 0
                    self.eyes_were_open = True

                    # Double blink detection
                    if self.last_blink_time != 0 and (current_time - self.last_blink_time) <= self.double_blink_window:
                        self.last_blink_time = 0
                        return "double_blink"

                    # Single blink
                    self.last_blink_time = current_time

                    if blink_duration >= 0.5:
                        blink_type = "long_blink"
                    else:
                        blink_type = "short_blink"

            return blink_type

        except (IndexError, AttributeError):
            return None

    
    def count_extended_fingers(self, hand_landmarks):
    # Orientation-aware, stable finger counting
        lm = hand_landmarks.landmark
        fingers_up = 0

        wrist = lm[0]
        index_mcp = lm[5]
        pinky_mcp = lm[17]

    # Determine hand orientation
        is_right_hand = index_mcp.x < pinky_mcp.x

    # --- Thumb ---
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        thumb_mcp = lm[2]

    # Use direction vector instead of fixed threshold
        thumb_vector_x = thumb_tip.x - thumb_mcp.x
        thumb_vector_y = abs(thumb_tip.y - thumb_mcp.y)

        if is_right_hand:
            if thumb_vector_x < -0.03 and thumb_vector_y < 0.08:
                fingers_up += 1
        else:
            if thumb_vector_x > 0.03 and thumb_vector_y < 0.08:
                fingers_up += 1

    # --- Other Fingers ---
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for tip, pip in zip(finger_tips, finger_pips):
            if lm[tip].y < lm[pip].y - 0.015:
                fingers_up += 1

        return fingers_up

    def detect_hand_gesture(self, hand_landmarks):
        lm = hand_landmarks.landmark

        # Thumb landmarks
        thumb_tip = lm[4]
        thumb_mcp = lm[2]

        # Other fingers tips and pips
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        # Fingers count
        fingers = self.count_extended_fingers(hand_landmarks)

        # Hand orientation
        is_right_hand = lm[5].x < lm[17].x

        # Check if all non-thumb fingers are folded
        non_thumb_folded = all(lm[tip].y > lm[pip].y for tip, pip in zip(finger_tips, finger_pips))

        # Thumb vertical vector
        thumb_vector_y = thumb_tip.y - thumb_mcp.y

        # --- Fist / Thumb gestures ---
        if non_thumb_folded:
            if thumb_vector_y < -0.03:
                return "thumbs_up"
            elif thumb_vector_y > 0.03:
                return "thumbs_down"
            else:
                return "fist"

        # --- Two fingers ---
        if fingers == 2:
            if lm[8].y < lm[6].y and lm[12].y < lm[10].y:
                return "two_fingers"

        # --- Three fingers ---
        if fingers == 3:
            return "three_fingers"

        # --- Palm open ---
        if fingers >= 4:
            wrist = lm[0]
            index_mcp = lm[5]
            pinky_tip = lm[20]
            palm_center_x = (index_mcp.x + pinky_tip.x) / 2
            tilt_x = palm_center_x - wrist.x
            tilt_threshold = 0.10

            if tilt_x < -tilt_threshold:
                return "tilt_left"
            elif tilt_x > tilt_threshold:
                return "tilt_right"
            else:
                return "palm_open"

        return "unknown"

    def get_frame_and_gestures(self):
        """Main detection loop with improved visualization"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None, None
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Enhanced preprocessing
        rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.1, beta=5)
        
        hand_results = hands.process(rgb_frame)
        hand_gesture = None
        hand_position = None
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_gesture = self.detect_hand_gesture(hand_landmarks)
                
                # Gesture confidence tracking
                if hand_gesture == self.last_confident_gesture:
                    self.gesture_confidence_counter += 1
                else:
                    self.gesture_confidence_counter = 0
                    self.last_confident_gesture = hand_gesture
                
                # Only add to history if confident
                if self.gesture_confidence_counter >= 2 or hand_gesture in ["thumbs_up", "fist", "tilt_left", "tilt_right"]:
                    self.gesture_history.append(hand_gesture)
                
                index_tip = hand_landmarks.landmark[8]
                frame_height, frame_width = frame.shape[:2]
                hand_position = (
                    int(index_tip.x * frame_width),
                    int(index_tip.y * frame_height)
                )
                
                self.hand_position_history.append(hand_position)
                if len(self.hand_position_history) >= 2:
                    avg_x = sum(p[0] for p in self.hand_position_history) / len(self.hand_position_history)
                    avg_y = sum(p[1] for p in self.hand_position_history) / len(self.hand_position_history)
                    hand_position = (int(avg_x), int(avg_y))
                
                # Enhanced visualization
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)
                )
                
                cv2.circle(frame, hand_position, 10, (0, 255, 255), -1)
                cv2.circle(frame, hand_position, 12, (255, 255, 255), 2)
                
                # Gesture indicator
                gesture_colors = {
                    "tilt_left": (255, 255, 0),
                    "tilt_right": (255, 255, 0),
                    "thumbs_up": (0, 255, 255),
                    "thumbs_down": (255, 100, 0),
                    "fist": (255, 0, 0),
                    "palm_open": (0, 255, 0)
                }
                gesture_color = gesture_colors.get(hand_gesture, (100, 100, 100))
                
                cv2.circle(frame, (50, 50), 35, gesture_color, -1)
                cv2.circle(frame, (50, 50), 37, (255, 255, 255), 3)
        
        face_results = face_mesh.process(rgb_frame)
        blink_type = None
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                blink_type = self.detect_blink_type(face_landmarks)
                
                # Visualize eyes
                LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
                
                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # EAR display
                if len(self.ear_history) > 0:
                    avg_ear = sum(self.ear_history) / len(self.ear_history)
                    ear_text = f"EAR: {avg_ear:.3f}"
                    cv2.putText(frame, ear_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Blink status
                blink_status = f"BLINK: {blink_type}" if blink_type else "Eyes Open"
                status_color = (0, 0, 255) if blink_type else (0, 255, 0)
                
                cv2.putText(frame, blink_status, (10, 65), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Emotion detection
        emotion_data = None
        current_time = time.time()
        if current_time - self.last_emotion_check > 1.5:
            emotion_data = self.emotion_detector.detect_emotion(frame)
            self.last_emotion_check = current_time
            
            if emotion_data:
                emotion_text = f"{emotion_data['emotion'].upper()} ({emotion_data['confidence']:.0f}%)"
                # Enhanced emotion display
                cv2.rectangle(frame, (5, 95), (240, 135), (0, 0, 0), -1)
                cv2.rectangle(frame, (5, 95), (240, 135), (0, 255, 0), 2)
                cv2.putText(frame, emotion_text, (15, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame, hand_gesture, blink_type, emotion_data, hand_position
    
    def get_smoothed_gesture(self):
        """Get most common recent gesture with confidence"""
        if not self.gesture_history:
            return None
        
        # Count gestures with recency bias
        gesture_counts = {}
        for i, g in enumerate(self.gesture_history):
            if g and g != "unknown":
                weight = math.exp((i - len(self.gesture_history)) * 0.2)
                gesture_counts[g] = gesture_counts.get(g, 0) + weight
        
        if gesture_counts:
            return max(gesture_counts.items(), key=lambda x: x[1])[0]
        return None
    
    def release(self):
        self.cap.release()


class Car:
    """Enhanced F1 sports car"""
    
    def __init__(self, x, y, color=RED):
        self.x = x
        self.y = y
        self.width = 70
        self.height = 120
        self.speed = 0
        self.max_speed = 20
        self.acceleration = 0.8
        self.friction = 0.3
        self.color = color
        self.angle = 0
        
        self.speed_lines = []
        self.exhaust_particles = []
        self.particle_system = ParticleSystem()
        self.pulse_timer = 0
        self.trail_points = deque(maxlen=20)
        
    def update_effects(self):
        """Update visual effects"""
        if self.speed > 8:
            for _ in range(int(self.speed / 4)):
                line = {
                    'x': self.x + random.randint(-30, 30),
                    'y': self.y + random.randint(-50, 50),
                    'length': random.randint(10, 30),
                    'alpha': 255
                }
                self.speed_lines.append(line)
        
        for line in self.speed_lines[:]:
            line['y'] += self.speed * 1.5
            line['alpha'] -= 20
            if line['alpha'] <= 0 or line['y'] > HEIGHT:
                self.speed_lines.remove(line)
        
        if self.speed > 10:
            self.particle_system.emit(self.x, self.y + self.height // 2, 'spark', 3, self.speed / 20)
        
        self.particle_system.update()
        self.pulse_timer = (self.pulse_timer + 1) % 60
        
        self.trail_points.append((self.x, self.y))
    
    def draw(self, surface):
        """Draw car with effects"""
        # Trail
        if len(self.trail_points) > 3:
            for i in range(1, len(self.trail_points)):
                alpha = int(100 * (i / len(self.trail_points)))
                try:
                    pygame.draw.line(surface, self.color, self.trail_points[i-1], self.trail_points[i], 2)
                except:
                    pass
        
        # Speed lines
        for line in self.speed_lines:
            color = (255, 255, 255, int(line['alpha']))
            s = pygame.Surface((3, line['length']), pygame.SRCALPHA)
            s.fill(color)
            surface.blit(s, (line['x'], line['y']))
        
        self.particle_system.draw(surface)
        
        # Car body
        body_points = [
            (self.x - self.width//2 + 5, self.y + self.height//2),
            (self.x + self.width//2 - 5, self.y + self.height//2),
            (self.x + self.width//2 - 8, self.y - self.height//2 + 15),
            (self.x - self.width//2 + 8, self.y - self.height//2 + 15)
        ]
        pygame.draw.polygon(surface, self.color, body_points)
        
        # Highlight
        highlight_color = tuple(min(255, c + 80) for c in self.color)
        pygame.draw.polygon(surface, highlight_color, [
            (self.x - self.width//3, self.y - self.height//4),
            (self.x + self.width//3, self.y - self.height//4),
            (self.x + self.width//3 - 5, self.y - self.height//6),
            (self.x - self.width//3 + 5, self.y - self.height//6)
        ])
        
        pygame.draw.polygon(surface, BLACK, body_points, 3)
        
        # Windshield
        windshield = [
            (self.x - self.width//3 + 5, self.y - self.height//4),
            (self.x + self.width//3 - 5, self.y - self.height//4),
            (self.x + self.width//3 - 8, self.y - self.height//2 + 20),
            (self.x - self.width//3 + 8, self.y - self.height//2 + 20)
        ]
        pygame.draw.polygon(surface, (120, 140, 180), windshield)
        pygame.draw.polygon(surface, (100, 120, 160), windshield, 2)
        
        # Wheels
        pulse = abs(math.sin(self.pulse_timer / 30 * math.pi))
        wheel_color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in DARK_GRAY)
        
        wheel_positions = [
            (self.x - self.width//3, self.y + self.height//3),
            (self.x + self.width//3, self.y + self.height//3),
            (self.x - self.width//3, self.y - self.height//4),
            (self.x + self.width//3, self.y - self.height//4)
        ]
        
        for wx, wy in wheel_positions:
            pygame.draw.circle(surface, BLACK, (int(wx), int(wy)), 11)
            pygame.draw.circle(surface, wheel_color, (int(wx), int(wy)), 8)
            pygame.draw.circle(surface, (100, 100, 100), (int(wx), int(wy)), 4)
        
        # Headlights
        brightness = min(255, 150 + int(self.speed * 5))
        headlight_color = (brightness, brightness, 0)
        
        hl_positions = [
            (self.x - self.width//4, self.y - self.height//2 + 8),
            (self.x + self.width//4, self.y - self.height//2 + 8)
        ]
        
        for hl_pos in hl_positions:
            GlowEffect.draw_neon_circle(surface, hl_pos, 5, headlight_color, glow_size=8)
        
        # Spoiler
        spoiler_color = tuple(max(0, c - 30) for c in self.color)
        pygame.draw.rect(surface, spoiler_color, 
                        (self.x - 3, self.y + self.height//2 - 5, 6, 15))
        pygame.draw.rect(surface, WHITE, 
                        (self.x - 3, self.y + self.height//2 - 5, 6, 15), 1)


class Obstacle:
    """Enhanced obstacles"""
    
    TYPES = ['cone', 'barrier', 'oil']
    
    def __init__(self, x, y, obs_type=None):
        self.x = x
        self.y = y
        self.width = 50
        self.height = 50
        self.speed = 5
        self.type = obs_type or random.choice(self.TYPES)
        self.rotation = 0
        self.particle_system = ParticleSystem()
        
    def update(self, game_speed):
        self.y += self.speed + game_speed
        self.rotation = (self.rotation + 5) % 360
        
        if random.random() < 0.1:
            self.particle_system.emit(self.x, self.y, 'smoke', 1, 0.5)
        
        self.particle_system.update()
        
    def draw(self, surface):
        self.particle_system.draw(surface)
        
        if self.type == 'cone':
            points = [
                (self.x, self.y - self.height//2),
                (self.x - self.width//2, self.y + self.height//2),
                (self.x + self.width//2, self.y + self.height//2)
            ]
            pygame.draw.polygon(surface, ORANGE, points)
            
            pygame.draw.polygon(surface, (255, 200, 0), [
                (self.x, self.y - self.height//2),
                (self.x - self.width//3, self.y + self.height//3),
                (self.x + self.width//3, self.y + self.height//3)
            ])
            pygame.draw.polygon(surface, BLACK, points, 2)
            
            for i in range(3):
                y = self.y - self.height//2 + i * 15
                pygame.draw.line(surface, WHITE, (self.x - 15, y), (self.x + 15, y), 3)
        
        elif self.type == 'barrier':
            pygame.draw.rect(surface, (200, 0, 0), 
                           (self.x - self.width//2, self.y - self.height//4, 
                            self.width, self.height//2))
            
            stripe_width = 8
            for i in range(0, self.width, stripe_width * 2):
                pygame.draw.rect(surface, WHITE, 
                               (self.x - self.width//2 + i, self.y - self.height//4,
                                stripe_width, self.height//2))
            
            pygame.draw.rect(surface, (255, 100, 100),
                           (self.x - self.width//2, self.y - self.height//4,
                            self.width, self.height//2), 3)
        
        elif self.type == 'oil':
            pygame.draw.ellipse(surface, (30, 30, 30),
                              (self.x - self.width//2, self.y - self.height//3,
                               self.width, self.height//1.5))
            
            pygame.draw.ellipse(surface, (80, 80, 80),
                              (self.x - self.width//3, self.y - self.height//4,
                               self.width//3, self.height//4))
    
    def is_off_screen(self):
        return self.y > HEIGHT + 50


class AudioManager:
    """Audio manager"""
    
    def __init__(self):
        self.sounds = {}
        try:
            self.sounds['boost'] = pygame.mixer.Sound('assets/turbo-flutter-336362.mp3')
            self.sounds['crash'] = pygame.mixer.Sound('assets/car-crash-sound-effect-376874.mp3')
        except:
            pass
        
        try:
            pygame.mixer.music.load('assets/background-music-224633.mp3')
            pygame.mixer.music.set_volume(0.4)
        except:
            pass
        
        volume_levels = {
            'boost': 0.6,
            'crash': 0.7,
        }
        
        for sound_name, volume in volume_levels.items():
            if sound_name in self.sounds:
                self.sounds[sound_name].set_volume(volume)
    
    def play_sound(self, sound_name):
        try:
            if sound_name in self.sounds:
                self.sounds[sound_name].play()
        except:
            pass
    
    def start_music(self):
        try:
            pygame.mixer.music.play(-1, fade_ms=2000)
        except:
            pass
    
    def stop_music(self):
        try:
            pygame.mixer.music.fadeout(1000)
        except:
            pass


class RaceMode:
    """Enhanced racing mode"""
    
    def __init__(self):
        self.car = Car(WIDTH // 2, HEIGHT - 150)
        
        self.obstacles = []
        self.score = 0
        self.high_score = 0
        self.game_speed = 4
        self.road_y = 0
        self.tree_positions = self.generate_trees()
        
        self.lane_width = 120
        self.num_lanes = 3
        road_center = WIDTH // 2
        self.lanes = [
            road_center - self.lane_width,
            road_center,
            road_center + self.lane_width
        ]
        
        self.current_lane = 1
        self.target_x = self.lanes[self.current_lane]
        
        self.lane_change_cooldown = 0
        self.lane_change_delay = 15
        
        self.current_emotion = None
        self.emotion_message = ""
        self.emotion_timer = 0
        self.last_emotion_time = time.time()
        self.emotion_commentary_cooldown = 6
        
        self.boost_active = False
        self.boost_timer = 0
        self.boost_particles = ParticleSystem()
        
        self.last_milestone_announced = 0
        self.milestones = [100, 200, 300, 500, 750, 1000]
        
        self.near_misses = 0
        self.last_near_miss_time = 0
        
        self.perfect_dodges = 0
        self.consecutive_perfect_dodges = 0
        self.crash_count = 0
        
        self.audio = AudioManager()
        self.audio.start_music()
        
        self.show_quit_dialog = False
        self.quit_dialog_selected = 0

    def generate_trees(self):
        trees = []
        for i in range(20):
            side = random.choice(['left', 'right'])
            x = 50 if side == 'left' else WIDTH - 50
            y = i * 100 + random.randint(-30, 30)
            trees.append({'x': x, 'y': y, 'type': random.choice(['tree', 'bush'])})
        return trees
    
    def spawn_obstacle(self):
        if random.random() < 0.015:
            lane = random.randint(0, self.num_lanes - 1)
            self.obstacles.append(Obstacle(self.lanes[lane], -80))
    
    def process_emotion_commentary(self, emotion_data):
        if not emotion_data:
            return
        
        current_time = time.time()
        
        if current_time - self.last_emotion_time < self.emotion_commentary_cooldown:
            return
        
        emotion = emotion_data.get('emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0)
        
        if confidence < 25:  # Lowered threshold
            return
        
        if emotion in ['angry', 'disgust']:
            emotion = 'angry'
        elif emotion in ['fear', 'sad']:
            emotion = 'sad'
        elif emotion == 'surprise':
            emotion = 'surprise'
        elif emotion not in ['happy']:
            emotion = 'focused'
        
        self.current_emotion = emotion
        self.emotion_timer = 200
        self.last_emotion_time = current_time
        
        commentary = self.generate_emotion_commentary(emotion, self.score)
        if commentary:
            speak(commentary)
            self.emotion_message = commentary
    
    def generate_emotion_commentary(self, emotion, score):
        if score < 30:
            level = "beginner"
        elif score < 50:
            level = "intermediate"
        elif score < 70:
            level = "advanced"
        else:
            level = "expert"
        
        commentary_lines = {
            'happy': {
                'beginner': [
                    f"Love that smile! {score} points!",
                    f"You're enjoying this! {score} points!",
                ],
                'intermediate': [
                    f"Brilliant! {score} points with a smile!",
                    f"{score} points! Happy drivers are fast!",
                ],
                'advanced': [
                    f"Incredible! {score} points with joy!",
                    f"Outstanding! {score}! Pure happiness!",
                ],
                'expert': [
                    f"LEGENDARY! {score} points of joy!",
                    f"ELITE! {score}! Unstoppable smile!",
                ]
            },
            'angry': {
                'beginner': [
                    f"Channel that fire! {score} points!",
                    f"{score}! Use that intensity!",
                ],
                'intermediate': [
                    f"Fierce! {score} points! Keep pushing!",
                    f"{score}! Your determination shows!",
                ],
                'advanced': [
                    f"Aggressive racing! {score} points!",
                    f"{score}! Control that power!",
                ],
                'expert': [
                    f"RUTHLESS! {score} points of rage!",
                    f"UNSTOPPABLE! {score}! Beast mode!",
                ]
            },
            'sad': {
                'beginner': [
                    f"You're doing well! {score} points!",
                    f"{score}! Keep your head up!",
                ],
                'intermediate': [
                    f"Stay strong! {score} is impressive!",
                    f"{score} shows real skill!",
                ],
                'advanced': [
                    f"Amazing at {score}! Believe in yourself!",
                    f"{score}! You're better than you think!",
                ],
                'expert': [
                    f"PHENOMENAL {score}! You're a champion!",
                    f"ELITE {score}! Be proud!",
                ]
            },
            'surprise': {
                'beginner': [
                    f"Wow! {score} already!",
                    f"Unexpected skills at {score}!",
                ],
                'intermediate': [
                    f"Amazing! {score} points! Shocking!",
                    f"Wow! {score}! Surpassing expectations!",
                ],
                'advanced': [
                    f"Unbelievable! {score} points!",
                    f"Astonishing {score}!",
                ],
                'expert': [
                    f"MIND-BLOWING! {score} points!",
                    f"INCREDIBLE {score}! Stunning!",
                ]
            },
            'focused': {
                'beginner': [
                    f"Great focus! {score} points!",
                    f"{score}! Steady driving!",
                ],
                'intermediate': [
                    f"Laser focus! {score} points!",
                    f"{score}! Professional!",
                ],
                'advanced': [
                    f"Elite concentration! {score}!",
                    f"{score}! Unbreakable focus!",
                ],
                'expert': [
                    f"ABSOLUTE FOCUS! {score}!",
                    f"LEGENDARY {score}! Ice cold!",
                ]
            }
        }
        
        if emotion in commentary_lines and level in commentary_lines[emotion]:
            return random.choice(commentary_lines[emotion][level])
        
        return None
    
    def check_near_miss(self):
        current_time = time.time()
        for obs in self.obstacles:
            dx = abs(self.car.x - obs.x)
            dy = abs(self.car.y - obs.y)
            
            if 40 < dx < 80 and dy < 100:
                if current_time - self.last_near_miss_time > 2:
                    self.near_misses += 1
                    self.last_near_miss_time = current_time
                    self.consecutive_perfect_dodges += 1
                    self.boost_particles.emit(self.car.x, self.car.y, 'spark', 10, 1.5)
                    
                    if self.current_emotion == 'happy':
                        comments = ["Close one! Great reflexes!"]
                    elif self.current_emotion == 'angry':
                        comments = ["Aggressive dodge!"]
                    elif self.current_emotion == 'sad':
                        comments = ["Great dodge! You're skilled!"]
                    else:
                        comments = ["Perfect timing!"]
                    
                    if random.random() < 0.4:
                        speak(random.choice(comments))
                    
                    return True
        return False
    
    def announce_milestone(self):
        for milestone in self.milestones:
            if self.score >= milestone and self.last_milestone_announced < milestone:
                self.last_milestone_announced = milestone
                
                base_messages = {
                    100: "ONE HUNDRED POINTS!",
                    200: "TWO HUNDRED POINTS!",
                    300: "THREE HUNDRED!",
                    500: "FIVE HUNDRED!",
                    750: "SEVEN FIFTY!",
                    1000: "ONE THOUSAND!"
                }
                
                message = base_messages.get(milestone, f"{milestone} points!")
                
                self.emotion_message = message
                self.emotion_timer = 250
                speak(message)
                return
    
    def update(self, gesture, emotion_data=None):
        if self.show_quit_dialog:
            if gesture == "thumbs_up":
                self.quit_dialog_selected = 1
                return "quit_game"
            elif gesture == "thumbs_down":
                self.show_quit_dialog = False
                self.quit_dialog_selected = 0
                speak("Continuing race!")
            return None
        
        if emotion_data:
            self.process_emotion_commentary(emotion_data)
        
        self.announce_milestone()
        self.check_near_miss()
        
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1
        
        if self.lane_change_cooldown == 0:
            if gesture == "tilt_left":
                if self.current_lane > 0:
                    self.current_lane -= 1
                    self.target_x = self.lanes[self.current_lane]
                    self.lane_change_cooldown = self.lane_change_delay
                    speak("Left lane")
            
            elif gesture == "tilt_right":
                if self.current_lane < self.num_lanes - 1:
                    self.current_lane += 1
                    self.target_x = self.lanes[self.current_lane]
                    self.lane_change_cooldown = self.lane_change_delay
                    speak("Right lane")
        
        if abs(self.car.x - self.target_x) > 2:
            self.car.x += (self.target_x - self.car.x) * 0.25
        else:
            self.car.x = self.target_x
        
        if gesture == "thumbs_up":
            if not self.boost_active:
                self.audio.play_sound('boost')
                self.boost_active = True
                self.boost_timer = 30
                speak("Turbo boost!")
            
            self.car.speed = min(self.car.max_speed, self.car.speed + self.car.acceleration * 2)
            self.game_speed = min(12, self.game_speed + 0.15)
            
        elif gesture == "fist":
            self.car.speed = max(0, self.car.speed - 1.5)
            self.game_speed = max(3, self.game_speed - 0.3)
            self.boost_active = False
            
        else:
            if self.car.speed > 5:
                self.car.speed -= self.car.friction * 0.5
            
            if self.boost_timer > 0:
                self.boost_timer -= 1
            else:
                self.boost_active = False
        
        if self.boost_active and self.boost_timer > 0:
            self.game_speed = min(15, self.game_speed + 0.1)
            self.boost_particles.emit(self.car.x + random.randint(-20, 20), 
                                     self.car.y + self.car.height//2, 'boost', 3, 1.2)
        
        self.boost_particles.update()
        self.car.update_effects()
        
        self.spawn_obstacle()
        for obs in self.obstacles[:]:
            obs.update(self.game_speed)
            if obs.is_off_screen():
                self.obstacles.remove(obs)
                self.score += 10
                
                if abs(self.car.x - obs.x) > 100:
                    self.perfect_dodges += 1
                
                if self.score > self.high_score:
                    self.high_score = self.score
        
        self.road_y += self.game_speed
        if self.road_y > 120:
            self.road_y = 0
        
        for tree in self.tree_positions:
            tree['y'] += self.game_speed
            if tree['y'] > HEIGHT + 50:
                tree['y'] = -50
                tree['x'] = 50 if tree['x'] < WIDTH//2 else WIDTH - 50
        
        if self.emotion_timer > 0:
            self.emotion_timer -= 1
        
        return None
    
    def check_collision(self):
        for obs in self.obstacles:
            dx = abs(self.car.x - obs.x)
            dy = abs(self.car.y - obs.y)
            
            if dx < (self.car.width + obs.width) / 2.5 and dy < (self.car.height + obs.height) / 2.5:
                self.audio.play_sound('crash')
                self.crash_count += 1
                self.consecutive_perfect_dodges = 0
                return True
        return False
    
    def draw_road(self, surface):
        # Sky gradient
        for i in range(HEIGHT // 3):
            color = (135 - i//10, 206 - i//10, 250 - i//15)
            pygame.draw.line(surface, color, (0, i), (WIDTH, i))
        
        # Grass
        grass_y = HEIGHT // 3
        pygame.draw.rect(surface, GRASS_GREEN, (0, grass_y, WIDTH, HEIGHT - grass_y))
        
        # Trees
        for tree in self.tree_positions:
            tx, ty = tree['x'], int(tree['y'])
            if grass_y < ty < HEIGHT:
                if tree['type'] == 'tree':
                    pygame.draw.rect(surface, (101, 67, 33), (tx - 8, ty - 20, 16, 40))
                    pygame.draw.circle(surface, (34, 139, 34), (tx, ty - 35), 30)
                    pygame.draw.circle(surface, (50, 180, 50), (tx - 15, ty - 40), 15)
                    pygame.draw.circle(surface, (50, 180, 50), (tx + 15, ty - 40), 15)
                else:
                    pygame.draw.circle(surface, (50, 180, 50), (tx, ty), 20)
        
        # Road
        road_width = 420
        road_x = (WIDTH - road_width) // 2
        pygame.draw.rect(surface, (60, 60, 60), (road_x, grass_y, road_width, HEIGHT - grass_y))
        
        # Road edges
        pygame.draw.rect(surface, WHITE, (road_x - 4, grass_y, 8, HEIGHT - grass_y))
        pygame.draw.rect(surface, WHITE, (road_x + road_width - 4, grass_y, 8, HEIGHT - grass_y))
        
        # Lane markings
        for lane_x in [WIDTH // 2 - self.lane_width, WIDTH // 2 + self.lane_width]:
            for i in range(-2, 15):
                y_pos = grass_y + (i * 80 + int(self.road_y)) % (HEIGHT - grass_y)
                if grass_y < y_pos < HEIGHT:
                    pygame.draw.rect(surface, YELLOW, (lane_x - 4, y_pos, 8, 50))
    
    def draw_gesture_guide(self, surface):
        """IMPROVED: Enhanced gesture guide"""
        guide_height = 80
        overlay = pygame.Surface((WIDTH, guide_height), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 220), (0, 0, WIDTH, guide_height))
        pygame.draw.rect(overlay, NEON_GREEN, (0, 0, WIDTH, guide_height), 4)
        surface.blit(overlay, (0, 0))
        
        font_title = pygame.font.Font(None, 32)
        font_gesture = pygame.font.Font(None, 26)
        
        title = font_title.render("🎮 CONTROLS", True, YELLOW)
        surface.blit(title, (25, 12))
        
        gestures = [
            ("🖐️ Tilt LEFT/RIGHT", "Lane Change", NEON_PINK),
            ("👍 Thumbs UP", "Boost", NEON_GREEN),
            ("✊ Fist", "Brake", RED),
            ("👀 Double Blink", "Quit", ORANGE)
        ]
        
        x_start = 25
        y = 48
        spacing = 300
        
        for i, (gesture, action, color) in enumerate(gestures):
            x = x_start + i * spacing
            
            gesture_text = font_gesture.render(gesture, True, color)
            action_text = font_gesture.render(f"= {action}", True, WHITE)
            
            surface.blit(gesture_text, (x, y))
            surface.blit(action_text, (x + 155, y))
    
    def draw_quit_dialog(self, surface):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 230), (0, 0, WIDTH, HEIGHT))
        surface.blit(overlay, (0, 0))
        
        dialog_width = 650
        dialog_height = 380
        dialog_x = (WIDTH - dialog_width) // 2
        dialog_y = (HEIGHT - dialog_height) // 2
        
        pygame.draw.rect(surface, DARK_GRAY, (dialog_x, dialog_y, dialog_width, dialog_height), border_radius=25)
        pygame.draw.rect(surface, NEON_PINK, (dialog_x, dialog_y, dialog_width, dialog_height), 6, border_radius=25)
        
        font_title = pygame.font.Font(None, 70)
        font_text = pygame.font.Font(None, 42)
        font_option = pygame.font.Font(None, 48)
        
        title = font_title.render("⚠️ QUIT GAME?", True, YELLOW)
        surface.blit(title, (dialog_x + dialog_width//2 - title.get_width()//2, dialog_y + 35))
        
        subtitle = font_text.render("Use gestures to choose:", True, WHITE)
        surface.blit(subtitle, (dialog_x + dialog_width//2 - subtitle.get_width()//2, dialog_y + 115))
        
        yes_box_y = dialog_y + 180
        no_box_y = dialog_y + 270
        box_height = 70
        
        pygame.draw.rect(surface, NEON_GREEN, (dialog_x + 60, yes_box_y, dialog_width - 120, box_height), border_radius=18)
        pygame.draw.rect(surface, WHITE, (dialog_x + 60, yes_box_y, dialog_width - 120, box_height), 4, border_radius=18)
        
        yes_text = font_option.render("👍 YES - Quit (Thumbs UP)", True, BLACK)
        surface.blit(yes_text, (dialog_x + dialog_width//2 - yes_text.get_width()//2, yes_box_y + 18))
        
        pygame.draw.rect(surface, RED, (dialog_x + 60, no_box_y, dialog_width - 120, box_height), border_radius=18)
        pygame.draw.rect(surface, WHITE, (dialog_x + 60, no_box_y, dialog_width - 120, box_height), 4, border_radius=18)
        
        no_text = font_option.render("👎 NO - Continue (Thumbs DOWN)", True, WHITE)
        surface.blit(no_text, (dialog_x + dialog_width//2 - no_text.get_width()//2, no_box_y + 18))
    
    def draw(self, surface):
        self.draw_road(surface)
        
        # Boost overlay
        if self.boost_active and self.boost_timer > 0:
            alpha = min(100, self.boost_timer * 3)
            boost_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            boost_surf.fill((255, 200, 0, alpha))
            surface.blit(boost_surf, (0, 0))
        
        for obs in self.obstacles:
            obs.draw(surface)
        
        self.boost_particles.draw(surface)
        self.car.draw(surface)
        
        self.draw_gesture_guide(surface)
        
        # IMPROVED HUD with modern design
        hud_width = 340
        hud_height = 200
        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        pygame.draw.rect(hud_surface, (0, 0, 0, 200), (0, 0, hud_width, hud_height), border_radius=20)
        pygame.draw.rect(hud_surface, NEON_BLUE, (0, 0, hud_width, hud_height), 3, border_radius=20)
        surface.blit(hud_surface, (25, 100))
        
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 30)
        
        score_text = font_large.render(f"Score: {self.score}", True, YELLOW)
        speed_text = font_medium.render(f"Speed: {int(self.car.speed * 10)} km/h", True, GREEN)
        lane_text = font_small.render(f"Lane: {self.current_lane + 1}/3", True, CYAN)
        
        surface.blit(score_text, (45, 115))
        surface.blit(speed_text, (45, 165))
        surface.blit(lane_text, (45, 210))
        
        # High score
        if self.high_score > 0:
            high_score_text = font_small.render(f"Best: {self.high_score}", True, NEON_PINK)
            surface.blit(high_score_text, (45, 250))
        
        # Speed meter
        meter_x = WIDTH - 160
        meter_y = 115
        meter_width = 130
        meter_height = 25
        
        pygame.draw.rect(surface, DARK_GRAY, (meter_x, meter_y, meter_width, meter_height), border_radius=12)
        speed_fill = int((self.car.speed / self.car.max_speed) * meter_width)
        
        if self.boost_active:
            speed_color = NEON_PINK
        elif self.car.speed < 12:
            speed_color = GREEN
        elif self.car.speed < 18:
            speed_color = ORANGE
        else:
            speed_color = RED
            
        pygame.draw.rect(surface, speed_color, (meter_x, meter_y, speed_fill, meter_height), border_radius=12)
        pygame.draw.rect(surface, speed_color, (meter_x, meter_y, meter_width, meter_height), 3, border_radius=12)
        
        if self.boost_active and self.boost_timer > 0:
            boost_font = pygame.font.Font(None, 42)
            boost_text = boost_font.render("🚀 BOOST!", True, NEON_PINK)
            surface.blit(boost_text, (WIDTH - 165, 150))
        
        # Emotion message
        if self.emotion_timer > 0:
            msg_font = pygame.font.Font(None, 52)
            msg_text = msg_font.render(self.emotion_message, True, PURPLE)
            
            msg_rect = msg_text.get_rect(center=(WIDTH // 2, 380))
            
            pulse = abs(math.sin(pygame.time.get_ticks() / 200))
            alpha = int(220 + 35 * pulse)
            
            glow_surface = pygame.Surface((msg_rect.width + 80, msg_rect.height + 50), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (255, 255, 255, alpha), glow_surface.get_rect(), border_radius=30)
            surface.blit(glow_surface, (msg_rect.x - 40, msg_rect.y - 25))
            
            pygame.draw.rect(surface, YELLOW, (msg_rect.x - 40, msg_rect.y - 25, 
                           msg_rect.width + 80, msg_rect.height + 50), 4, border_radius=30)
            
            shadow_text = msg_font.render(self.emotion_message, True, (100, 0, 100))
            surface.blit(shadow_text, (msg_rect.x + 4, msg_rect.y + 4))
            
            surface.blit(msg_text, msg_rect)
        
        # Emotion indicator
        if self.current_emotion:
            emotion_font = pygame.font.Font(None, 32)
            emotion_icons = {
                'happy': '😊',
                'angry': '😤',
                'sad': '😢',
                'focused': '🎯',
                'surprise': '😲'
            }
            icon = emotion_icons.get(self.current_emotion, '😐')
            
            emotion_bg = pygame.Surface((220, 60), pygame.SRCALPHA)
            pygame.draw.rect(emotion_bg, (0, 0, 0, 180), (0, 0, 220, 60), border_radius=15)
            pygame.draw.rect(emotion_bg, NEON_BLUE, (0, 0, 220, 60), 3, border_radius=15)
            surface.blit(emotion_bg, (WIDTH - 240, 190))
            
            emotion_text = emotion_font.render(f"Mood: {icon} {self.current_emotion.title()}", True, CYAN)
            surface.blit(emotion_text, (WIDTH - 228, 205))
        
        if self.show_quit_dialog:
            self.draw_quit_dialog(surface)


class Game:
    """Main game controller"""
    
    def __init__(self):
        self.state = "home"
        self.gesture_detector = GestureDetector()
        self.race_mode = RaceMode()
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.selected_menu_item = 0
        self.menu_items = ["Start Race", "Exit"]
        
        self.audio = AudioManager()
        self.menu_particle_system = ParticleSystem()
        
        # FIXED: Track game over state for blink restart
        self.game_over_state = False
        self.game_over_timer = 0

    def draw_home_screen(self):
        """IMPROVED: Enhanced home screen"""
        # Animated gradient background
        for i in range(HEIGHT):
            ratio = i / HEIGHT
            r = int(40 + ratio * 120)
            g = int(80 + ratio * 170)
            b = int(180 + ratio * 75)
            pygame.draw.line(screen, (r, g, b), (0, i), (WIDTH, i))
        
        # Animated title
        time_offset = pygame.time.get_ticks() / 1000
        font_title = pygame.font.Font(None, 110)
        title_text = "🏎️ F1 GESTURE RACER"
        title = font_title.render(title_text, True, BLUE)
        
        title_y = 70 + math.sin(time_offset) * 12
        
        # Glow effect
        for i in range(20, 0, -1):
            alpha = int(40 * (1 - i / 20))
            s = pygame.Surface((title.get_width() + i * 4, title.get_height() + i * 2), pygame.SRCALPHA)
            s.fill((255, 220, 0, alpha))
            screen.blit(s, (WIDTH // 2 - title.get_width() // 2 - i * 2, title_y - i))
        
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, title_y))
        
        
        
        # Animated car
        car_x = WIDTH // 2
        car_y = 240
        home_car = Car(car_x, car_y, RED)
        home_car.draw(screen)
        
        # IMPROVED Menu options
        font_menu = pygame.font.Font(None, 65)
        menu_y_start = 330
        
        menu_options = [
            ("🏁 START RACE", "Blink Once", GREEN),
            ("❌ EXIT GAME", "Double Blink", RED)
        ]
        
        for i, (text, hint, color) in enumerate(menu_options):
            y = menu_y_start + i * 130
            
            bg_width = 500
            bg_height = 100
            bg_x = WIDTH // 2 - bg_width // 2
            bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
            
            # Animated selection
            if i == self.selected_menu_item:
                pulse = abs(math.sin(pygame.time.get_ticks() / 300))
                alpha = int(150 + 100 * pulse)
                pygame.draw.rect(bg_surface, (*color, alpha), (0, 0, bg_width, bg_height), border_radius=25)
                pygame.draw.rect(bg_surface, color, (0, 0, bg_width, bg_height), 5, border_radius=25)
            else:
                pygame.draw.rect(bg_surface, (255, 255, 255, 80), (0, 0, bg_width, bg_height), border_radius=25)
                pygame.draw.rect(bg_surface, (150, 150, 150), (0, 0, bg_width, bg_height), 2, border_radius=25)
            
            screen.blit(bg_surface, (bg_x, y - 15))
            
            option_text = font_menu.render(text, True, color if i == self.selected_menu_item else WHITE)
            screen.blit(option_text, (WIDTH // 2 - option_text.get_width() // 2, y))
            
            hint_font = pygame.font.Font(None, 32)
            hint_text = hint_font.render(hint, True, WHITE if i == self.selected_menu_item else LIGHT_GRAY)
            screen.blit(hint_text, (WIDTH // 2 - hint_text.get_width() // 2, y + 58))
        
        # IMPROVED Controls guide
        guide_x = 60
        guide_y = HEIGHT - 220
        guide_width = 600
        
        guide_surface = pygame.Surface((guide_width, 200), pygame.SRCALPHA)
        pygame.draw.rect(guide_surface, (0, 0, 0, 220), (0, 0, guide_width, 200), border_radius=20)
        pygame.draw.rect(guide_surface, NEON_GREEN, (0, 0, guide_width, 200), 4, border_radius=20)
        screen.blit(guide_surface, (guide_x, guide_y))
        
        guide_font = pygame.font.Font(None, 30)
        guide_title = pygame.font.Font(None, 38).render("🎮 Racing Controls", True, YELLOW)
        screen.blit(guide_title, (guide_x + 20, guide_y + 15))
        
        controls = [
            "🖐️  Tilt palm LEFT/RIGHT = Lane change",
            "👍 Thumbs UP = Turbo Boost",
            "✊  Fist = Brake",
            "😊 Your emotions = Live commentary!"
        ]
        
        for i, ctrl in enumerate(controls):
            text = guide_font.render(ctrl, True, WHITE)
            screen.blit(text, (guide_x + 25, guide_y + 60 + i * 35))
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state != "home":
                        self.state = "home"
                        self.game_over_state = False
                    else:
                        self.running = False
    
    def run(self):
        """Main game loop"""
        print("\n" + "="*70)
        print("🏎️  F1 GESTURE RACER PRO - ENHANCED EDITION")
        print("="*70)
        print("\n✅ FIXED FEATURES:")
        print("   🎭 IMPROVED emotion detection (better accuracy)")
        print("   👋 ENHANCED hand gesture recognition")
        print("   👁️  FIXED blink controls (single blink to restart)")
        print("   🎨 MODERN GUI with smooth animations")
        print("   🎤 Real-time voice commentary")
        print("\n✅ CONTROLS:")
        print("   🖐️  Tilt palm LEFT/RIGHT to change lanes")
        print("   👍 Thumbs UP for turbo boost")
        print("   👎 Thumbs DOWN to decline quit")
        print("   ✊ Fist to brake")
        print("   👀 Single BLINK to start race")
        print("   👀👀 Double BLINK to open quit menu")
        print("   😊 Smile = Happy commentary!")
        print("   😤 Angry face = Aggressive commentary!")
        print("   😢 Sad face = Encouraging commentary!")
        print("\n✅ All systems ready!")
        print("="*70 + "\n")
        
        last_emotion_data = None
        
        while self.running:
            self.clock.tick(60)
            self.handle_events()
            
            frame, hand_gesture, blink_type, emotion_data, hand_position = self.gesture_detector.get_frame_and_gestures()
            smoothed_gesture = self.gesture_detector.get_smoothed_gesture()
            
            if emotion_data:
                last_emotion_data = emotion_data
            
            # Handle blink events
            if blink_type:
                # FIXED: Single blink to start/restart
                if blink_type == "short_blink":
                    if self.state == "home":
                        self.state = "race"
                        self.race_mode = RaceMode()
                        self.game_over_state = False
                        speak("Let's race! Show me your emotions!")
                    elif self.game_over_state:
                        # Restart from game over
                        self.race_mode = RaceMode()
                        self.game_over_state = False
                        speak("New race started!")
                
                # Double blink for quit
                elif blink_type == "double_blink":
                    if self.state == "race" and not self.game_over_state:
                        if not self.race_mode.show_quit_dialog:
                            self.race_mode.show_quit_dialog = True
                            speak("Do you want to quit? Thumbs up for yes, thumbs down for no.")
                    elif self.state == "home":
                        self.running = False
                        speak("Goodbye!")
            
            if self.state == "home":
                self.draw_home_screen()
            
            elif self.state == "race":
                if not self.game_over_state:
                    result = self.race_mode.update(smoothed_gesture, last_emotion_data)
                    
                    # Check if user wants to quit from dialog
                    if result == "quit_game":
                        speak("Returning to menu!")
                        self.state = "home"
                        self.race_mode = RaceMode()
                    else:
                        self.race_mode.draw(screen)
                        
                        # Check collision
                        if self.race_mode.check_collision():
                            self.game_over_state = True
                            self.game_over_timer = 0
                            final_score = self.race_mode.score
                            emotion = self.race_mode.current_emotion or 'focused'
                            
                            # Emotion-based game over commentary
                            if emotion == 'happy':
                                if final_score > 500:
                                    speak(f"Amazing! {final_score} points with joy! You're incredible!")
                                elif final_score > 300:
                                    speak(f"Great job! {final_score} with a smile!")
                                else:
                                    speak(f"Nice effort! {final_score} points! Keep that smile!")
                            
                            elif emotion == 'angry':
                                if final_score > 500:
                                    speak(f"Incredible! {final_score} points! Pure power!")
                                elif final_score > 300:
                                    speak(f"Fierce! {final_score}! Channel that energy!")
                                else:
                                    speak(f"Good effort! {final_score}! Stay focused!")
                            
                            elif emotion == 'sad':
                                if final_score > 500:
                                    speak(f"See? {final_score} points! You're amazing!")
                                elif final_score > 300:
                                    speak(f"Great work! {final_score}! Believe in yourself!")
                                else:
                                    speak(f"You got {final_score}! Don't be discouraged!")
                            
                            else:
                                if final_score > 500:
                                    speak(f"Excellent! {final_score} points! Elite performance!")
                                elif final_score > 300:
                                    speak(f"Great run! {final_score} points!")
                                else:
                                    speak(f"Final score: {final_score}. Good effort!")
                
                # IMPROVED: Game over screen
                if self.game_over_state:
                    self.game_over_timer += 1
                    
                    # Draw game over overlay
                    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    pygame.draw.rect(overlay, (0, 0, 0, 200), (0, 0, WIDTH, HEIGHT))
                    screen.blit(overlay, (0, 0))
                    
                    final_score = self.race_mode.score
                    emotion = self.race_mode.current_emotion or 'focused'
                    
                    font_big = pygame.font.Font(None, 90)
                    font_med = pygame.font.Font(None, 55)
                    font_small = pygame.font.Font(None, 38)
                    
                    # GAME OVER title
                    game_over = font_big.render("GAME OVER", True, RED)
                    screen.blit(game_over, (WIDTH // 2 - game_over.get_width() // 2, HEIGHT // 2 - 180))
                    
                    # Score
                    score_text = font_med.render(f"Final Score: {final_score}", True, YELLOW)
                    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2 - 90))
                    
                    # Stats
                    stats = []
                    if self.race_mode.perfect_dodges > 0:
                        stats.append(f"Perfect Dodges: {self.race_mode.perfect_dodges} 🎯")
                    if self.race_mode.near_misses > 0:
                        stats.append(f"Near Misses: {self.race_mode.near_misses} 😰")
                    
                    # Rank based on score and emotion
                    emotion_emojis = {
                        'happy': '😊',
                        'angry': '😤',
                        'sad': '😢',
                        'surprise': '😲',
                        'focused': '🎯'
                    }
                    emotion_emoji = emotion_emojis.get(emotion, '😐')
                    
                    if final_score > 1000:
                        rank = f"💎 {emotion_emoji} EXPERT RACER!"
                        rank_color = PURPLE
                    elif final_score > 500:
                        rank = f"🔥 {emotion_emoji} ADVANCED DRIVER!"
                        rank_color = ORANGE
                    elif final_score > 200:
                        rank = f"⚡ {emotion_emoji} SKILLED RACER!"
                        rank_color = GREEN
                    else:
                        rank = f"🌟 {emotion_emoji} GREAT START!"
                        rank_color = CYAN
                    
                    rank_text = font_med.render(rank, True, rank_color)
                    screen.blit(rank_text, (WIDTH // 2 - rank_text.get_width() // 2, HEIGHT // 2 - 20))
                    
                    # Display stats
                    y_offset = HEIGHT // 2 + 50
                    for stat in stats:
                        stat_text = font_small.render(stat, True, WHITE)
                        screen.blit(stat_text, (WIDTH // 2 - stat_text.get_width() // 2, y_offset))
                        y_offset += 42
                    
                    # FIXED: Restart instruction
                    if self.game_over_timer > 60:  # Show after 1 second
                        restart_font = pygame.font.Font(None, 48)
                        restart = restart_font.render("👁️ BLINK ONCE to Race Again", True, NEON_GREEN)
                        
                        # Pulsing effect
                        pulse = abs(math.sin(pygame.time.get_ticks() / 400))
                        alpha = int(200 + 55 * pulse)
                        restart_surface = pygame.Surface((restart.get_width() + 40, restart.get_height() + 20), pygame.SRCALPHA)
                        pygame.draw.rect(restart_surface, (0, 255, 0, alpha // 3), restart_surface.get_rect(), border_radius=15)
                        screen.blit(restart_surface, (WIDTH // 2 - restart.get_width() // 2 - 20, y_offset + 20 - 10))
                        
                        screen.blit(restart, (WIDTH // 2 - restart.get_width() // 2, y_offset + 20))
                        
                        # Back to menu option
                        menu_text = font_small.render("or press ESC for menu", True, LIGHT_GRAY)
                        screen.blit(menu_text, (WIDTH // 2 - menu_text.get_width() // 2, y_offset + 75))
            
            # Camera preview - IMPROVED design
            if frame is not None:
                preview_width, preview_height = 260, 195
                frame_resized = cv2.resize(frame, (preview_width, preview_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                
                preview_x = WIDTH - preview_width - 25
                preview_y = HEIGHT - preview_height - 25
                
                # Border with glow
                pygame.draw.rect(screen, BLACK, (preview_x - 5, preview_y - 5, 
                                                preview_width + 10, preview_height + 10), border_radius=10)
                pygame.draw.rect(screen, NEON_GREEN, (preview_x - 3, preview_y - 3, 
                                                preview_width + 6, preview_height + 6), 4, border_radius=10)
                screen.blit(frame_surface, (preview_x, preview_y))
                
                # Gesture indicator
                if smoothed_gesture:
                    gesture_font = pygame.font.Font(None, 28)
                    gesture_bg = pygame.Surface((preview_width, 40), pygame.SRCALPHA)
                    pygame.draw.rect(gesture_bg, (0, 0, 0, 220), (0, 0, preview_width, 40), border_radius=8)
                    pygame.draw.rect(gesture_bg, NEON_GREEN, (0, 0, preview_width, 40), 3, border_radius=8)
                    screen.blit(gesture_bg, (preview_x, preview_y + preview_height - 40))
                    
                    gesture_colors = {
                        "tilt_left": YELLOW,
                        "tilt_right": YELLOW,
                        "thumbs_up": NEON_PINK,
                        "thumbs_down": ORANGE,
                        "fist": RED,
                        "palm_open": GREEN
                    }
                    gesture_color = gesture_colors.get(smoothed_gesture, WHITE)
                    
                    gesture_text = gesture_font.render(f"✋ {smoothed_gesture.upper()}", True, gesture_color)
                    screen.blit(gesture_text, (preview_x + 15, preview_y + preview_height - 32))
            
            pygame.display.flip()
        
        self.gesture_detector.release()
        pygame.quit()
        print("\n👋 Thanks for playing!")


if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Game interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()