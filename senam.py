import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Label, Button, StringVar
from PIL import Image, ImageTk
import mediapipe as mp
import math
import tkinter.messagebox as messagebox
import threading
import time

class MotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Detector")
        self.root.configure(bg="#222831")
        self.root.geometry("1100x700")  # Lebar diperbesar untuk list gerakan
        self.root.resizable(False, False)

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background="#222831", foreground="#eeeeee", font=('Segoe UI', 14))
        style.configure('TButton', font=('Segoe UI', 12), padding=6)
        style.configure('TCheckbutton', background="#222831", foreground="#eeeeee", font=('Segoe UI', 11))

        # Frame utama
        main_frame = tk.Frame(root, bg="#222831")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Frame kiri (kamera & kontrol)
        left_frame = tk.Frame(main_frame, bg="#222831")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame kanan (list gerakan)
        right_frame = tk.Frame(main_frame, bg="#222831")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(30, 0))

        # Judul
        self.title_label = Label(left_frame, text="Deteksi Gerakan Senam", font=('Segoe UI', 22, 'bold'), bg="#222831", fg="#00adb5")
        self.title_label.pack(pady=(0, 10))

        # Video frame
        self.video_frame = Label(left_frame, bg="#393e46", width=800, height=450, relief=tk.RIDGE, bd=3)
        self.video_frame.pack(pady=(0, 15))

        # Status label
        self.label = Label(left_frame, text="Gerakan Belum Terdeteksi", font=('Segoe UI', 15), bg="#222831", fg="#eeeeee")
        self.label.pack(pady=(0, 10))

        # Checkbox deteksi jari
        self.finger_detection_enabled = tk.BooleanVar(value=True)
        self.finger_checkbox = tk.Checkbutton(
            left_frame, text="Aktifkan Deteksi Jari", variable=self.finger_detection_enabled,
            font=('Segoe UI', 11), bg="#222831", fg="#eeeeee", selectcolor="#00adb5", activebackground="#222831"
        )
        self.finger_checkbox.pack(pady=5)

        # Tombol kalibrasi ulang
        self.recalib_button = Button(left_frame, text="Kalibrasi Ulang", command=self.recalibrate, height=1, bg="#00adb5", fg="#222831", font=('Segoe UI', 12, 'bold'), relief=tk.RAISED, bd=2, activebackground="#393e46")
        self.recalib_button.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Tombol keluar
        self.button = Button(left_frame, text="Keluar", command=self.stop_app, height=2, bg="#393e46", fg="#eeeeee", font=('Segoe UI', 12, 'bold'), relief=tk.RAISED, bd=2, activebackground="#00adb5")
        self.button.pack(fill=tk.X, padx=10, pady=(0, 10))

        # --- List Gerakan ---
        Label(right_frame, text="Daftar Gerakan", font=('Segoe UI', 16, 'bold'), bg="#222831", fg="#00adb5").pack(pady=(10, 10))
        self.movement_list = [
            ("Kepala Angguk ke Atas", "Up"),
            ("Kepala Angguk ke Bawah", "Down"),
            ("Kepala Geleng ke Kiri", "Left"),
            ("Kepala Geleng ke Kanan", "Right"),
            ("Kepala Miring ke Kiri", "TiltLeft"),
            ("Kepala Miring ke Kanan", "TiltRight"),
            ("Kedua Tangan Direntangkan", "BothArmsExtended"),
            ("Tangan Kanan Direntangkan", "LeftArmExtended"),
            ("Tangan Kiri Direntangkan", "RightArmExtended"),
            ("Kedua Tangan Diangkat", "BothHandsRaised"),
            ("Tangan Kanan Diangkat", "LeftHandRaised"),
            ("Tangan Kiri Diangkat", "RightHandRaised"),
            ("Kedua Tangan Menyilang ke Kiri Bawah", "LeftCrossRight"),
            ("Kedua Tangan Menyilang ke Kanan Bawah", "RightCrossLeft"),
        ]
        self.movement_status = {}
        self.movement_vars = {}
        for label, key in self.movement_list:
            var = StringVar()
            var.set("❌ " + label)
            lbl = Label(right_frame, textvariable=var, font=('Segoe UI', 13), bg="#222831", anchor="w", fg="#eeeeee", padx=10)
            lbl.pack(fill=tk.X, pady=2)
            self.movement_vars[key] = var
            self.movement_status[key] = False

        # --- Timer untuk setiap gerakan (dalam frame) ---
        self.movement_hold_frames = {key: 0 for _, key in self.movement_list}
        self.movement_hold_target = 8 * 30  # 8 detik * 30 fps

        # Add this new attribute for tracking shown popups
        self.popup_shown = {key: False for _, key in self.movement_list}

        # Initialize camera
        self.capture = cv2.VideoCapture(0)

        # MediaPipe initialization with higher sensitivity
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=2
        )

        self.mp_drawing = mp.solutions.drawing_utils

        # Head movement tracking with additional hold detection
        self.prev_nose_y = None
        self.prev_nose_x = None
        self.neutral_nose_x = None
        self.neutral_nose_y = None
        self.calibration_frames = 0
        self.calibration_needed = True
        self.calibration_threshold = 30  # Number of frames to establish neutral position

        # Position thresholds
        self.head_movement_threshold = 0.05  # Threshold to detect intentional movement
        self.hold_threshold = 0.04  # Threshold to determine if position is being held

        # Hold state tracking
        self.current_hold_state = "Neutral"
        self.hold_frames = 0
        self.hold_frames_threshold = 15  # Frames required to consider a position "held"
        self.hold_positions = {
            "Left": 0,
            "Right": 0,
            "Up": 0,
            "Down": 0,
            "Neutral": 0,
            "TiltLeft": 0,
            "TiltRight": 0
        }

        # Arm position tracking
        self.arm_positions = {
            "BothArmsExtended": 0,
            "LeftArmExtended": 0,
            "RightArmExtended": 0,
            "ArmsDown": 0,
            "BothHandsRaised": 0,
            "RightHandRaised": 0,
            "LeftHandRaised": 0,
            "RightCrossLeft": 0,
            "LeftCrossRight": 0,
            "RightCrossRight": 0,
            "LeftCrossLeft": 0,
            "ArmsCrossed": 0
        }
        self.arm_hold_threshold = 15  # Frames to consider arm position held
        self.arm_extension_threshold = 0.15  # How far arm needs to be from body
        self.hand_raise_threshold = 0.2  # Threshold for raised hands (vertical position)

        # Hand gesture tracking
        self.finger_count_history = []
        self.history_length = 5

        self.update_frame()

    def calculate_angle(self, a, b, c):
        """Menghitung sudut (dalam derajat) di titik b dari tiga titik a-b-c."""
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def show_popup(self, message):
        """Show popup in a separate thread"""
        def popup():
            popup_window = tk.Toplevel(self.root)
            popup_window.overrideredirect(True)  # Remove window decorations
            
            # Calculate position (center of screen)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            width = 300
            height = 100
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            
            popup_window.geometry(f"{width}x{height}+{x}+{y}")
            popup_window.configure(bg="#00adb5")
            
            # Add message
            label = Label(popup_window, text=message, 
                         font=('Segoe UI', 12, 'bold'),
                         bg="#00adb5", fg="white",
                         wraplength=280)
            label.pack(expand=True)
            
            # Close after 3 seconds
            self.root.after(3000, popup_window.destroy)
        
        # Run in main thread to avoid tkinter threading issues
        self.root.after(0, popup)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose and hands
        results_pose = self.pose.process(frame_rgb)
        results_hands = self.hands.process(frame_rgb)

        # Reset detection flags to prevent overlap
        motion_detected = False
        head_gesture_active = False
        arm_gesture_active = False
        hand_gesture_active = False

        # Track active gesture type for priority
        active_gesture_type = None
        active_gesture_frames = 0
        active_gesture_text = ""
        active_gesture_display = ""

        # Head and arm position detection with hold functionality
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

            # Extract key arm landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            # Current positions
            current_nose_x = nose.x
            current_nose_y = nose.y

            # Calibration phase - establish neutral position
            if self.calibration_needed:
                if self.calibration_frames < self.calibration_threshold:
                    if self.neutral_nose_x is None:
                        self.neutral_nose_x = current_nose_x
                        self.neutral_nose_y = current_nose_y
                    else:
                        # Slowly adjust to account for slight movements during calibration
                        self.neutral_nose_x = 0.9 * self.neutral_nose_x + 0.1 * current_nose_x
                        self.neutral_nose_y = 0.9 * self.neutral_nose_y + 0.1 * current_nose_y

                    self.calibration_frames += 1
                    self.label.config(text=f"Kalibrasi... {self.calibration_frames}/{self.calibration_threshold}")
                    cv2.putText(frame, f"KALIBRASI: {self.calibration_frames}/{self.calibration_threshold}",
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    self.calibration_needed = False
                    self.label.config(text="Kalibrasi Selesai - Siap Deteksi")
            else:
                # Calculate position relative to neutral
                x_diff = current_nose_x - self.neutral_nose_x
                y_diff = current_nose_y - self.neutral_nose_y

                horizontal_strength = abs(x_diff)
                vertical_strength = abs(y_diff)
                movement_ratio = horizontal_strength / (vertical_strength + 1e-6)
                strong_threshold = 0.08

                # Ambil landmark telinga
                left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
                # Ambil dagu (gunakan landmark mulut bawah jika dagu tidak ada)
                chin = landmarks[self.mp_pose.PoseLandmark.MOUTH_BOTTOM] if hasattr(self.mp_pose.PoseLandmark, "MOUTH_BOTTOM") else None

                # Hitung kemiringan kepala (selisih y telinga)
                ear_y_diff = left_ear.y - right_ear.y
                tilt_threshold = 0.04  # threshold kemiringan, bisa disesuaikan

                # --- Tambahan deteksi mendongak ---
                # Jika hidung lebih tinggi dari rata-rata telinga, anggap mendongak
                avg_ear_y = (left_ear.y + right_ear.y) / 2
                look_up_threshold = 0.03  # threshold sensitifitas mendongak

                current_position = "Neutral"
                if nose.y < avg_ear_y - look_up_threshold:
                    current_position = "Up"
                elif abs(ear_y_diff) > tilt_threshold:
                    if ear_y_diff > 0:
                        current_position = "TiltRight"  # Kepala miring ke kanan (telinga kiri lebih rendah)
                    else:
                        current_position = "TiltLeft"   # Kepala miring ke kiri (telinga kanan lebih rendah)
                elif movement_ratio > 1.5 and horizontal_strength > strong_threshold:
                    if x_diff > 0:
                        current_position = "Right"
                    else:
                        current_position = "Left"
                elif movement_ratio < 0.67 and vertical_strength > strong_threshold:
                    if y_diff > 0:
                        current_position = "Down"
                    else:
                        current_position = "Up"

                # Update head position hold counters
                for position in self.hold_positions:
                    if position == current_position:
                        self.hold_positions[position] += 1
                    else:
                        self.hold_positions[position] = 0

                # Detect arm positions
                # Calculate horizontal and vertical distances for arms
                left_arm_horizontal = abs(left_wrist.x - left_shoulder.x)
                right_arm_horizontal = abs(right_wrist.x - right_shoulder.x)
                left_arm_vertical = abs(left_wrist.y - left_shoulder.y)
                right_arm_vertical = abs(right_wrist.y - right_shoulder.y)

                # Determine current arm position
                left_arm_extended = left_arm_horizontal > self.arm_extension_threshold and left_wrist.y < left_shoulder.y + 0.1
                right_arm_extended = right_arm_horizontal > self.arm_extension_threshold and right_wrist.y < right_shoulder.y + 0.1

                # Detect raised hands (vertical position)
                left_hand_raised = left_wrist.y < left_shoulder.y - self.hand_raise_threshold
                right_hand_raised = right_wrist.y < right_shoulder.y - self.hand_raise_threshold

                # Calculate elbow angles
                right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

                current_arm_position = "ArmsDown"
                crossed_threshold = 0.10  # threshold jarak antara pergelangan tangan (bisa disesuaikan)
                wrist_distance = np.sqrt(
                    (left_wrist.x - right_wrist.x) ** 2 +
                    (left_wrist.y - right_wrist.y) ** 2
                )
                # Cek apakah kedua pergelangan tangan saling berdekatan dan berada di antara bahu
                if (
                    wrist_distance < crossed_threshold and
                    min(left_shoulder.x, right_shoulder.x) < left_wrist.x < max(left_shoulder.x, right_shoulder.x) and
                    min(left_shoulder.x, right_shoulder.x) < right_wrist.x < max(left_shoulder.x, right_shoulder.x)
                ):
                    current_arm_position = "ArmsCrossed"
                elif (
                    right_wrist.x < left_shoulder.x and
                    right_wrist.y > left_shoulder.y + 0.05 and
                    abs(right_wrist.y - left_shoulder.y) > 0.10 and
                    abs(right_wrist.x - left_shoulder.x) > 0.05 and
                    right_elbow_angle < 60
                ):
                    current_arm_position = "RightCrossLeft"
                elif (
                    left_wrist.x > right_shoulder.x and
                    left_wrist.y > right_shoulder.y + 0.05 and
                    abs(left_wrist.y - right_shoulder.y) > 0.10 and
                    abs(left_wrist.x - right_shoulder.x) > 0.05 and
                    left_elbow_angle < 60
                ):
                    current_arm_position = "LeftCrossRight"
                elif (
                    right_wrist.x > right_shoulder.x and
                    right_wrist.y > right_shoulder.y + 0.05 and
                    abs(right_wrist.y - right_shoulder.y) > 0.10 and
                    abs(right_wrist.x - right_shoulder.x) > 0.05 and
                    right_elbow_angle < 60
                ):
                    current_arm_position = "RightCrossRight"
                elif (
                    left_wrist.x < left_shoulder.x and
                    left_wrist.y > left_shoulder.y + 0.05 and
                    abs(left_wrist.y - left_shoulder.y) > 0.10 and
                    abs(left_wrist.x - left_shoulder.x) > 0.05 and
                    left_elbow_angle < 60
                ):
                    current_arm_position = "LeftCrossLeft"
                elif left_hand_raised and right_hand_raised:
                    current_arm_position = "BothHandsRaised"
                elif left_hand_raised:
                    current_arm_position = "LeftHandRaised"
                elif right_hand_raised:
                    current_arm_position = "RightHandRaised"
                elif left_arm_extended and right_arm_extended:
                    current_arm_position = "BothArmsExtended"
                elif left_arm_extended:
                    current_arm_position = "LeftArmExtended"
                elif right_arm_extended:
                    current_arm_position = "RightArmExtended"

                # Debug info untuk sudut siku
                cv2.putText(frame, f"Right Elbow Angle: {right_elbow_angle:.1f}", (400, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(frame, f"Left Elbow Angle: {left_elbow_angle:.1f}", (400, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # Update arm position hold counters
                for position in self.arm_positions:
                    if position == current_arm_position:
                        self.arm_positions[position] += 1
                    else:
                        self.arm_positions[position] = 0

                # Show debug info for arms
                cv2.putText(frame, f"Left arm H: {left_arm_horizontal:.3f}", (50, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Right arm H: {right_arm_horizontal:.3f}", (50, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Arm Position: {current_arm_position}", (50, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Check for held head positions
                for position, frames in self.hold_positions.items():
                    if frames >= self.hold_frames_threshold and position != "Neutral":
                        if frames > active_gesture_frames:  # Higher frame count gets priority
                            head_gesture_active = True
                            active_gesture_frames = frames
                            active_gesture_type = "Head"

                            # Prepare text for display
                            hold_duration = frames / 30.0  # Convert frames to seconds (assuming 30fps)

                            if position in ["Left", "Right"]:
                                active_gesture_text = f"Kepala Geleng ke {position} (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"GELENG {position.upper()}: {hold_duration:.1f}s"
                            elif position in ["TiltLeft", "TiltRight"]:
                                active_gesture_text = f"Kepala Miring ke {position} (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"MIRING {position.upper()}: {hold_duration:.1f}s"
                            else:
                                active_gesture_text = f"Kepala Angguk ke {position} (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"ANGGUK {position.upper()}: {hold_duration:.1f}s"

                # Check for held arm positions - only if no head gesture is active or arm gesture has more frames
                for position, frames in self.arm_positions.items():
                    if frames >= self.arm_hold_threshold and position != "ArmsDown":
                        if frames > active_gesture_frames:  # Higher frame count gets priority
                            arm_gesture_active = True
                            active_gesture_frames = frames
                            active_gesture_type = "Arm"

                            # Prepare text for display
                            hold_duration = frames / 30.0  # Convert frames to seconds (assuming 30fps)

                            if position == "BothArmsExtended":
                                active_gesture_text = f"Kedua Tangan Direntangkan (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"KEDUA TANGAN DIRENTANGKAN: {hold_duration:.1f}s"
                            elif position == "LeftArmExtended":
                                active_gesture_text = f"Tangan Kanan Direntangkan (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"TANGAN KANAN DIRENTANGKAN: {hold_duration:.1f}s"
                            elif position == "RightArmExtended":
                                active_gesture_text = f"Tangan Kiri Direntangkan (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"TANGAN KIRI DIRENTANGKAN: {hold_duration:.1f}s"
                            elif position == "BothHandsRaised":
                                active_gesture_text = f"Kedua Tangan Diangkat (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"KEDUA TANGAN DIANGKAT: {hold_duration:.1f}s"
                            elif position == "LeftHandRaised":
                                active_gesture_text = f"Tangan Kanan Diangkat (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"TANGAN KANAN DIANGKAT: {hold_duration:.1f}s"
                            elif position == "RightHandRaised":
                                active_gesture_text = f"Tangan Kiri Diangkat (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"TANGAN KIRI DIANGKAT: {hold_duration:.1f}s"
                            elif position == "LeftCrossRight":
                                active_gesture_text = f"Kedua Tangan Menyilang ke Kiri Bawah (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"KEDUA TANGAN SILANG KIRI BAWAH: {hold_duration:.1f}s"
                            elif position == "RightCrossLeft":
                                active_gesture_text = f"Kedua Tangan Menyilang ke Kanan Bawah (Tahan: {hold_duration:.1f}s)"
                                active_gesture_display = f"KEDUA TANGAN SILANG KANAN BAWAH: {hold_duration:.1f}s"

                # Show debug info on frame for head position
                cv2.putText(frame, f"X-diff: {x_diff:.3f}", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Y-diff: {y_diff:.3f}", (50, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Head Position: {current_position}", (50, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Display active gesture (if any)
                if head_gesture_active or arm_gesture_active:
                    motion_detected = True
                    self.label.config(text=active_gesture_text)
                    cv2.putText(frame, active_gesture_display,
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 255, 0) if active_gesture_type == "Head" else (0, 0, 255), 2)
                else:
                    if self.hold_positions["Neutral"] >= self.hold_frames_threshold:
                        self.label.config(text="Tahan Posisi untuk Deteksi")

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame_rgb, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Enhanced hand detection with history smoothing
        if self.finger_detection_enabled.get() and results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks,
                                                  results_hands.multi_handedness):
                hand_type = handedness.classification[0].label
                fingers = self.count_fingers(hand_landmarks, hand_type)

                # Add to history for smoothing
                self.finger_count_history.append(fingers)
                if len(self.finger_count_history) > self.history_length:
                    self.finger_count_history.pop(0)

                # Use mode (most common value) for stability
                smoothed_fingers = max(set(self.finger_count_history),
                                     key=self.finger_count_history.count)

                motion_detected = True
                if smoothed_fingers > 0:
                    self.label.config(text=f"{hand_type}: {smoothed_fingers} Jari")
                    # Visual feedback
                    cv2.putText(frame, f"{hand_type.upper()} HAND: {smoothed_fingers} fingers",
                               (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    self.label.config(text=f"{hand_type}: Kepalkan Tangan")

                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        elif not self.finger_detection_enabled.get():
            # Jika deteksi jari dimatikan, tampilkan info
            cv2.putText(frame, "Deteksi Jari: OFF", (50, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        if not motion_detected and not self.calibration_needed:
            self.label.config(text="Tahan Posisi Kepala untuk Deteksi")

        # Tambahkan update untuk checklist gerakan
        for label, key in self.movement_list:
            # Cek head gesture
            if key in self.hold_positions and self.hold_positions[key] >= self.hold_frames_threshold:
                self.movement_hold_frames[key] += 1
            # Cek arm gesture
            elif key in self.arm_positions and self.arm_positions[key] >= self.arm_hold_threshold:
                self.movement_hold_frames[key] += 1
            else:
                self.movement_hold_frames[key] = 0

            # Jika sudah 8 detik, checklist
            if (self.movement_hold_frames[key] >= self.movement_hold_target 
                and not self.movement_status[key] 
                and not self.popup_shown[key]):
                
                self.movement_vars[key].set("✅ " + label)
                self.movement_status[key] = True
                self.popup_shown[key] = True
                
                # Show success popup
                popup_message = f"Berhasil!\nGerakan '{label}' telah dilakukan dengan benar!"
                self.show_popup(popup_message)
                
            elif not self.movement_status[key]:
                self.movement_vars[key].set("❌ " + label)

        # Convert frame to PhotoImage for Tkinter
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video frame
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        # Schedule the next update
        self.root.after(10, self.update_frame)

    def count_fingers(self, hand_landmarks, hand_type):
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]
        mcp_ids = [2, 5, 9, 13, 17]  # Added for more precise detection
        fingers_up = 0

        # More sensitive thumb detection
        thumb_tip = hand_landmarks.landmark[tip_ids[0]]
        thumb_mcp = hand_landmarks.landmark[mcp_ids[0]]

        # Different logic based on hand type
        if hand_type == "Right":
            if thumb_tip.x < thumb_mcp.x - 0.05:  # More sensitive threshold
                fingers_up += 1
        else:
            if thumb_tip.x > thumb_mcp.x + 0.05:  # More sensitive threshold
                fingers_up += 1

        # More sensitive finger detection using multiple joints
        for i in range(1, 5):
            tip = hand_landmarks.landmark[tip_ids[i]]
            pip = hand_landmarks.landmark[pip_ids[i]]
            mcp = hand_landmarks.landmark[mcp_ids[i]]

            # Finger is up if tip is above both PIP and MCP joints
            if tip.y < min(pip.y, mcp.y) - 0.02:  # Added small buffer
                fingers_up += 1

        return fingers_up

    def recalibrate(self):
        """Reset kalibrasi posisi kepala."""
        self.calibration_needed = True
        self.calibration_frames = 0
        self.neutral_nose_x = None
        self.neutral_nose_y = None
        self.label.config(text="Kalibrasi Ulang Dimulai...")

    def stop_app(self):
        self.capture.release()
        self.pose.close()
        self.hands.close()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = MotionDetectorApp(root)
    root.mainloop()