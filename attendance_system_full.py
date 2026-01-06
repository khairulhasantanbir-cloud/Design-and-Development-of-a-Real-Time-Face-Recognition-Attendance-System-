

import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
import face_recognition  # type: ignore
from PIL import Image, ImageTk  # type: ignore
import tkinter as tk
from tkinter import messagebox

from simple_facerec import SimpleFacerec


def eye_aspect_ratio(eye: List[Tuple[int, int]]) -> float:
    """Compute the eye aspect ratio for a list of six landmark points."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return float((A + B) / (2.0 * C)) if C > 0 else 0.0


def run_dual_confirmation_session(
    duration_minutes: int = 40,
    checkout_window_minutes: int = 5,
) -> Tuple[Dict[str, List[datetime]], List[str]]:
    
    start_time = datetime.now()
    window_end = start_time + timedelta(minutes=duration_minutes)
    final_end = window_end + timedelta(minutes=checkout_window_minutes)
    print(
        f"Automatic attendance session started at {start_time.strftime('%H:%M:%S')} "
        f"with a required second confirmation after {duration_minutes} minutes."
    )
    # Initialise face recogniser and roster
    sfr = SimpleFacerec()
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    sfr.load_encoding_images(images_dir)
    roster = sfr.known_face_names.copy()

    # Dictionaries to hold per‑person calibration and blink state
    baseline_ear: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, 0.0))
    blink_threshold: Dict[str, float] = {}
    blink_state: Dict[str, Tuple[bool, int]] = defaultdict(lambda: (False, 0))
    blink_times: Dict[str, List[datetime]] = defaultdict(list)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the camera. Check camera index or permissions.")

    try:
        while True:
            now = datetime.now()
            if now > final_end:
                print("Session ended. Proceeding to evaluation and manual review...")
                break
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            face_locations, face_names = sfr.detect_known_faces(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
            for (face_loc, name, landmarks) in zip(face_locations, face_names, landmarks_list):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
                cv2.putText(
                    frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2
                )
                if name == "Unknown" or name not in roster:
                    continue
                # Ensure both eyes landmarks are present
                if "left_eye" not in landmarks or "right_eye" not in landmarks:
                    continue
                left_ear = eye_aspect_ratio(landmarks["left_eye"])
                right_ear = eye_aspect_ratio(landmarks["right_eye"])
                ear = (left_ear + right_ear) / 2.0
                # Calibration: compute baseline EAR for each person
                if name not in blink_threshold:
                    count, total = baseline_ear[name]
                    # Use only frames where eyes are reasonably open for calibration
                    if ear > 0.18:
                        baseline_ear[name] = (count + 1, total + ear)
                        count += 1
                        total += ear
                    if count >= 15:
                        avg_ear = total / count
                        blink_threshold[name] = avg_ear * 0.5
                        print(
                            f"Calibrated {name}: baseline EAR {avg_ear:.3f}, "
                            f"blink threshold {blink_threshold[name]:.3f}"
                        )
                    continue
                # Blink detection
                blink_detected, closed_frames = blink_state[name]
                threshold = blink_threshold[name]
                eye_closed = ear < threshold
                if eye_closed:
                    closed_frames += 1
                else:
                    if closed_frames >= 1:
                        blink_detected = True
                    closed_frames = 0
                blink_state[name] = (blink_detected, closed_frames)
                if blink_detected:
                    blink_times[name].append(now)
                    blink_state[name] = (False, 0)
                    # Print check‑in or additional confirmation
                    if len(blink_times[name]) == 1:
                        print(f"{name} checked in at {now.strftime('%H:%M:%S')}")
                    else:
                        print(f"{name} additional confirmation at {now.strftime('%H:%M:%S')}")
            # Display status text
            remaining = (final_end - now).total_seconds()
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            if blink_threshold:
                instruction = (
                    f"Blink to confirm (second confirmation after {duration_minutes} min). "
                    f"Time left: {minutes:02d}:{seconds:02d}"
                )
            else:
                instruction = (
                    f"Calibrating... keep eyes open. "
                    f"Time left: {minutes:02d}:{seconds:02d}"
                )
            cv2.putText(
                frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.imshow("Automatic Attendance Session", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Session terminated early. Proceeding to manual review...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return blink_times, roster


def evaluate_auto_status(
    roster: List[str],
    blink_times: Dict[str, List[datetime]],
    duration_minutes: int,
    tolerance_seconds: int = 60,
) -> Dict[str, str]:
   
    statuses: Dict[str, str] = {}
    required_seconds = duration_minutes * 60 - tolerance_seconds
    for name in roster:
        times = blink_times.get(name, [])
        if len(times) >= 2:
            times_sorted = sorted(times)
            diff = (times_sorted[-1] - times_sorted[0]).total_seconds()
            if diff >= required_seconds:
                statuses[name] = "Present"
                continue
        statuses[name] = "Absent"
    return statuses


def show_manual_dashboard(
    statuses: Dict[str, str],
    output_filename: str,
    blink_times: Dict[str, List[datetime]],
) -> None:
    """
    Display a GUI to allow manual adjustment of attendance statuses.

    Parameters
    ----------
    statuses: Dict[str, str]
        A dictionary of student names to automatic status ("Present" or "Absent").
    output_filename: str
        The path of the CSV file to write when saving.
    blink_times: Dict[str, List[datetime]]
        Recorded blink times for each student; used to populate check‑in and
        check‑out times when saving.
    """
    root = tk.Tk()
    root.title("Attendance Review Dashboard")

    # Scrollable canvas and frame for many students
    canvas = tk.Canvas(root, borderwidth=0)
    frame = tk.Frame(canvas)
    vsb = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((4, 4), window=frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", on_frame_configure)

    # Load images and build rows
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    students = []
    for file in sorted(os.listdir(images_dir)):
        name, ext = os.path.splitext(file)
        if ext.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        img_path = os.path.join(images_dir, file)
        students.append((name, img_path))
    if not students:
        messagebox.showerror("No Images", "No student images found in the 'images' directory.")
        root.destroy()
        return

    photo_refs: List[ImageTk.PhotoImage] = []
    status_vars: Dict[str, tk.StringVar] = {}
    for row, (name, img_path) in enumerate(students):
        image = Image.open(img_path)
        image.thumbnail((80, 80))
        photo = ImageTk.PhotoImage(image)
        photo_refs.append(photo)
        lbl_img = tk.Label(frame, image=photo)
        lbl_img.grid(row=row, column=0, padx=5, pady=5)
        lbl_name = tk.Label(frame, text=name, font=("Arial", 10, "bold"))
        lbl_name.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        # Default to auto status if present, else absent
        default_status = statuses.get(name, "Absent")
        status_var = tk.StringVar(value=default_status)
        status_vars[name] = status_var
        rb_present = tk.Radiobutton(frame, text="Present", variable=status_var, value="Present")
        rb_absent = tk.Radiobutton(frame, text="Absent", variable=status_var, value="Absent")
        rb_present.grid(row=row, column=2, padx=5)
        rb_absent.grid(row=row, column=3, padx=5)

    def save_final():
        date_str = datetime.now().strftime("%Y-%m-%d")
        # Write header if file doesn't exist
        new_file = not os.path.exists(output_filename)
        with open(output_filename, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["Name", "CheckInTime", "CheckOutTime", "Date", "Status"])
            for name in status_vars:
                status = status_vars[name].get()
                times = blink_times.get(name, [])
                first_time = times[0].strftime("%H:%M:%S") if times else ""
                final_time = times[-1].strftime("%H:%M:%S") if times else ""
                writer.writerow([name, first_time, final_time, date_str, status])
        messagebox.showinfo("Saved", f"Attendance saved to {output_filename}")
        root.destroy()

    save_button = tk.Button(root, text="Save Attendance", command=save_final)
    save_button.pack(pady=10)
    root.mainloop()


def main() -> None:
    """Run the full attendance system with automatic session and manual dashboard."""
    duration_minutes = 40
    checkout_window_minutes = 5
    blink_times, roster = run_dual_confirmation_session(duration_minutes, checkout_window_minutes)
    # Evaluate automatic statuses
    auto_statuses = evaluate_auto_status(roster, blink_times, duration_minutes)
    # Output file for final attendance
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_filename = f"attendance_full_{date_str}.csv"
    # Show manual dashboard for review and final save
    show_manual_dashboard(auto_statuses, output_filename, blink_times)


if __name__ == "__main__":
    main()
