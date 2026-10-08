"""
Complete Face Recognition System for Attendance Marking

This module implements a working face recognition system using face_recognition library.
It encodes student faces during registration and matches them in classroom images.
"""

from email.mime import image
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import cv2
import face_recognition
import numpy as np
from pathlib import Path
import csv
from PIL import Image, ImageDraw
from config import (STUDENTS_FOLDER, 
                    CLASSROOM_FOLDER,
                    MONGODB_URI,
                    DATABASE_NAME,
                    FACE_MATCH_TOLERANCE,
                    UNKNOWN_FACE_THRESHOLD,
                    MIN_CONFIDENCE_MARGIN,
                    FACE_DETECTION_MODEL,
                    MIN_MATCHES_FOR_PRESENT,
                    FACE_DETECTION_WORKERS)
from pymongo import MongoClient



try:
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

    Image = None
    cv2 = None


client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

MIN_FACE_DETECTIONS = 2
MIN_PRESENT_VOTES = 2
MIN_AGREEING_ENCODINGS = 2

def encode_student_faces(roll_number: str) -> bool:
    """
    Encode all face images for a student and store in database
    
    Args:
        roll_number: Student's roll number
    
    Returns:
        bool: True if encoding successful, False otherwise
    """
    if not FACE_RECOGNITION_AVAILABLE:
        print("Error: face_recognition library not available. Cannot encode faces.")
        return False
    
    student_folder = STUDENTS_FOLDER / roll_number
    
    if not student_folder.exists():
        return False
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
        image_files.extend(student_folder.glob(ext))
    
    if len(image_files) < 4:
        return False
    
    all_encodings = []
    
    for image_path in image_files:
        try:
            image = face_recognition.load_image_file(str(image_path))
            face_locations = face_recognition.face_locations(image, model='hog')
            
            if len(face_locations) > 0:
                encodings = face_recognition.face_encodings(image, face_locations)
                all_encodings.extend(encodings)
                
        except Exception as e:
            print(f"Error encoding face from {image_path}: {e}")
            continue
    
    if len(all_encodings) < MIN_FACE_DETECTIONS:
        print(f"Warning: Only found {len(all_encodings)} face(s) for {roll_number}, need at least {MIN_FACE_DETECTIONS}")
        return False
    
    # if len(all_encodings) > 1: 27-07-26
    #     avg_encoding = np.mean(all_encodings, axis=0)
    # else:
    #     avg_encoding = all_encodings[0]
    
    # encoding_list = avg_encoding.tolist()
    encoding_list = [e.tolist() for e in all_encodings]
    
    db.face_encodings.update_one(
        {'roll_number': roll_number},
        {
            '$set': {
                'roll_number': roll_number,
                'encoding': encoding_list,
                'num_images': len(image_files),
                'num_faces_detected': len(all_encodings)
            }
        },
        upsert=True
    )
    
    print(f"Successfully encoded faces for {roll_number} from {len(image_files)} images")
    return True


# def load_all_student_encodings() -> Dict[str, np.ndarray]: 24-07-26
#     encodings_dict = {}
    
#     try:
#         for record in db.face_encodings.find({}):
#             roll_number = record['roll_number']
#             encoding_array = np.array(record['encoding'])
#             encodings_dict[roll_number] = encoding_array
        
#     except Exception as e:
#         print(f"Error loading encodings: {e}")
    
    # return encodings_dict
def load_all_student_encodings() -> Dict[str, List[np.ndarray]]:
    """roll_number -> list of encoding vectors (NOT a single averaged vector)."""
    encodings_dict = {}
    try:
        for record in db.face_encodings.find({}):
            roll_number = record['roll_number']
            vecs = record.get('encodings')
            if vecs:
                encodings_dict[roll_number] = [np.array(v) for v in vecs]
            elif record.get('encoding'):  # backward-compat for un-migrated docs
                encodings_dict[roll_number] = [np.array(record['encoding'])]
    except Exception as e:
        print(f"Error loading encodings: {e}")
    return encodings_dict
# def load_encodings_for_roll_numbers(roll_numbers: List[str]) -> Dict[str, np.ndarray]:
#     encodings_dict = {}
#     try:
#         records = db.face_encodings.find({'roll_number': {'$in': roll_numbers}})
#         for record in records:
#             encodings_dict[record['roll_number']] = np.array(record['encoding'])
#     except Exception as e:
#         pass
#     return encodings_dict
def load_encodings_for_roll_numbers(roll_numbers: List[str]) -> Dict[str, List[np.ndarray]]:
    encodings_dict = {}
    try:
        records = db.face_encodings.find({'roll_number': {'$in': roll_numbers}})
        for record in records:
            vecs = record.get('encodings')
            if vecs:
                encodings_dict[record['roll_number']] = [np.array(v) for v in vecs]
            elif record.get('encoding'):
                encodings_dict[record['roll_number']] = [np.array(record['encoding'])]
    except Exception:
        pass
    return encodings_dict

def match_probe_to_students(probe_encoding, student_encodings: Dict[str, List[np.ndarray]],
                             tolerance: float, min_agreeing: int = MIN_AGREEING_ENCODINGS):
    """
    Compare one detected face against every stored photo-encoding of every
    student. A student is only a candidate if at least `min_agreeing` of
    THEIR OWN stored photos are within tolerance of this probe face — a
    single lucky match against one photo is not enough to accuse someone.

    Returns (best_roll_or_None, best_distance_or_None, second_best_distance_or_None).
    The second-best is returned so callers can apply an ambiguity margin
    (as match_faces() already does) on top of this.
    """
    candidates = []  # (roll, min_distance, agreeing_count)

    for roll, enc_list in student_encodings.items():
        if not enc_list:
            continue
        stacked = np.stack(enc_list)
        distances = face_recognition.face_distance(stacked, probe_encoding)
        agreeing = int(np.sum(distances <= tolerance))
        if agreeing < min_agreeing:
            continue
        candidates.append((roll, float(np.min(distances)), agreeing))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda c: c[1])  # by distance, ascending
    best_roll, best_dist, _ = candidates[0]
    second_best_dist = candidates[1][1] if len(candidates) > 1 else None
    return best_roll, best_dist, second_best_dist



def detect_faces_in_image(image_path: str):
    if not FACE_RECOGNITION_AVAILABLE:
        return [], []
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)
        if len(face_locations) == 0:
            return [], []
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return face_encodings, face_locations
    except Exception as e:
        print(f"Error detecting faces in {image_path}: {e}")
        return [], []


def match_faces(classroom_encodings: List[np.ndarray],
                student_encodings: Dict[str, np.ndarray]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Best-match assignment of classroom faces to registered students with unknown rejection.

    Args:
        classroom_encodings: List of face encodings from classroom images
        student_encodings: Dictionary of student roll_number -> encoding

    Returns:
        (matches, metrics):
          - matches: roll_number -> number of accepted matches
          - metrics: {'unknown_faces': int, 'ambiguous_matches': int, 'weak_matches': int}
    """
    matches: Dict[str, int] = {}
    metrics = {
        'unknown_faces': 0,
        'ambiguous_matches': 0,
        'weak_matches': 0,
    }

    if not classroom_encodings or not student_encodings:
        return matches, metrics

    rolls = list(student_encodings.keys())
    student_vecs = np.stack([student_encodings[r] for r in rolls]) if rolls else np.empty((0, 128))

    for classroom_encoding in classroom_encodings:
        try:
            distances = face_recognition.face_distance(student_vecs, classroom_encoding)
        except Exception:
            distances = np.array([
                face_recognition.face_distance([student_encodings[r]], classroom_encoding)[0]
                for r in rolls
            ])

        if distances.size == 0:
            metrics['unknown_faces'] += 1
            continue

        # Find best and second-best matches
        order = np.argsort(distances)
        best_idx = order[0]
        best_roll = rolls[best_idx]
        best_dist = distances[best_idx]
        second_best_dist = distances[order[1]] if distances.size > 1 else None

        # Reject if distance exceeds match tolerance
        if best_dist > FACE_MATCH_TOLERANCE:
            metrics['unknown_faces'] += 1
            continue

        # If second best is too close, mark ambiguous and skip
        if second_best_dist is not None and (second_best_dist - best_dist) < MIN_CONFIDENCE_MARGIN:
            metrics['ambiguous_matches'] += 1
            continue

        # Weak match warning zone (optional): near unknown threshold
        if best_dist > UNKNOWN_FACE_THRESHOLD:
            metrics['weak_matches'] += 1
            # Still reject to avoid false positives
            metrics['unknown_faces'] += 1
            continue

        # Accept best match
        matches[best_roll] = matches.get(best_roll, 0) + 1

    return matches, metrics


# def mark_attendance_from_classroom_images(class_id, folder_path, session_id, tolerance=0.6):
#     print("STEP 1: Loading encodings from DB")

#     student_encodings = load_all_student_encodings()
#     if not student_encodings:
#         print("No encodings found")
#         return {"attendance": {}}

#     attendance = {roll: "absent" for roll in student_encodings.keys()}

#     rolls = list(student_encodings.keys())
#     student_vecs = np.array(list(student_encodings.values()))

#     folder = Path(folder_path)
#     image_files = sorted(
#         [img for img in folder.glob("*") if img.suffix.lower() in [".jpg", ".jpeg", ".png"]],
#         key=lambda x: x.stat().st_mtime,
#         reverse=True
#     )[:5]

#     if not image_files:
#         print("No classroom images found")
#         return {"attendance": attendance}

#     for img_file in image_files:
#         print("Processing:", img_file.name)
#         image = face_recognition.load_image_file(str(img_file))

#         # resize for better detection
#         small_img = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

#         # use CNN model for accuracy
#         face_locations = face_recognition.face_locations(small_img, model="cnn")
#         encodings = face_recognition.face_encodings(small_img, face_locations)

#         print(f"[DEBUG] Faces detected: {len(encodings)}")

#         for face_encoding in encodings:
#             distances = face_recognition.face_distance(student_vecs, face_encoding)
#             best_idx = np.argmin(distances)
#             best_roll = rolls[best_idx]
#             best_dist = distances[best_idx]

#             print(f" Match: {best_roll}, Distance: {best_dist:.4f}")

#             if best_dist < tolerance:
#                 attendance[best_roll] = "present"
#                 print(f" PRESENT: {best_roll}")
#             else:
#                 print("No match")

#     print("FINAL:", attendance)
#     return {"attendance": attendance}

# def mark_attendance_from_classroom_images(class_id, folder_path, session_id, tolerance=0.6): 24-07-26
#     print("STEP 1: Loading encodings from DB")

#     student_encodings = load_all_student_encodings()
#     if not student_encodings:
#         print("No encodings found")
#         return {"attendance": {}}

#     attendance = {roll: "absent" for roll in student_encodings.keys()}
#     rolls = list(student_encodings.keys())
#     student_vecs = np.array(list(student_encodings.values()))

#     folder = Path(folder_path)
#     image_files = sorted(
#         [img for img in folder.glob("*") if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
#          and not img.name.startswith("annotated_")],  # skip already-annotated images
#         key=lambda x: x.stat().st_mtime,
#         reverse=True
#     )[:5]

#     if not image_files:
#         print("No classroom images found")
#         return {"attendance": attendance}

#     for img_file in image_files:
#         print("Processing:", img_file.name)

#         # Load image via face_recognition (returns RGB numpy array)
#         image = face_recognition.load_image_file(str(img_file))
#         small_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

#         face_locations = face_recognition.face_locations(small_img, model="cnn")
#         encodings = face_recognition.face_encodings(small_img, face_locations)

#         print(f"[DEBUG] Faces detected: {len(encodings)}")

#         # Load a SEPARATE copy of the image for drawing
#         # cv2 loads as BGR, face_recognition loads as RGB
#         # We use cv2 here because cv2.rectangle and cv2.putText work on BGR numpy arrays
#         draw_img = cv2.imread(str(img_file))
#         if draw_img is None:
#             import numpy as np
#             from PIL import Image as PILImage
#             pil_img = PILImage.open(str(img_file)).convert('RGB')
#             draw_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#             print(f"[ANNOTATE] Used PIL fallback for: {img_file.name}")

#         if draw_img is None:
#             print(f"[ANNOTATE] SKIPPING - could not load image: {img_file}")
#             continue
#         # face_locations gives coordinates at 0.5x scale (because we resized)
#         # so we must multiply coordinates by 2 to map back to original image size
#         scale = 2  # because fx=0.5 earlier

#         # zip encodings with face_locations so we know WHERE each face is
#         for face_encoding, face_location in zip(encodings, face_locations):
#             top, right, bottom, left = face_location

#             # Scale coordinates back to original image size
#             top    = int(top    * scale)
#             right  = int(right  * scale)
#             bottom = int(bottom * scale)
#             left   = int(left   * scale)

#             # Compare this face against all known students
#             # distances = face_recognition.face_distance(student_vecs, face_encoding)

#             # matched_roll = None
#             # for idx, dist in enumerate(distances):
#             #     roll = rolls[idx]
#             #     if dist < tolerance:
#             #         attendance[roll] = "present"
#             #         matched_roll = roll  # remember who matched
#             #         print(f"PRESENT: {roll}, Distance: {dist:.4f}")
#             assignment = assign_faces_to_students(encodings, face_locations, student_vecs, rolls, tolerance)

#         for i, (face_encoding, face_location) in enumerate(zip(encodings, face_locations)):
#             top, right, bottom, left = face_location
#             top    = int(top    * scale)
#             right  = int(right  * scale)
#             bottom = int(bottom * scale)
#             left   = int(left   * scale)

#             matched_roll = assignment.get(i)
#             if matched_roll:
#                 # GREEN box — face was recognised
#                 color = (0, 255, 0)
#                 label = matched_roll
#             else:
#                 # RED box — face detected but not recognised
#                 color = (0, 0, 255)
#                 label = "Unknown"

#             # Draw rectangle around face
#             # cv2.rectangle(image, top-left corner, bottom-right corner, BGR color, thickness)
#             cv2.rectangle(draw_img, (left, top), (right, bottom), color, 2)

#             # Draw filled rectangle behind text so label is readable
#             cv2.rectangle(draw_img, (left, top - 25), (right, top), color, -1)

#             # Put student roll number or "Unknown" as label above the box
#             cv2.putText(draw_img, label, (left + 4, top - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#         # Save the annotated image with "annotated_" prefix
#         # Your professor routes already look for files starting with "annotated_"
#         # so this is what makes them show up in the edit_attendance page
#         annotated_path = folder / f"annotated_{img_file.name}"
#         success = cv2.imwrite(str(annotated_path), draw_img)
#         if success:
#             print(f"[ANNOTATED] Saved: {annotated_path.name}")
#         else:
#             # Fallback: use PIL to save
#             try:
#                 rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
#                 PILImage.fromarray(rgb).save(str(annotated_path))
#                 print(f"[ANNOTATED] Saved via PIL fallback: {annotated_path.name}")
#             except Exception as e:
#                 print(f"[ANNOTATED] FAILED to save: {e}")

#     print("FINAL:", attendance)
#     return {"attendance": attendance}

def mark_attendance_from_classroom_images(class_id, folder_path, session_id, tolerance=None,
                                           roll_numbers: List[str] = None):
    tolerance = tolerance or FACE_MATCH_TOLERANCE
    student_encodings = load_all_student_encodings()
    if roll_numbers is not None:
        student_encodings = {r: v for r, v in student_encodings.items() if r in roll_numbers}
    rolls = list(student_encodings.keys())

    if not student_encodings:
        print("No encodings found")
        return {"attendance": {}}

    # First pass: get the voted attendance decision across all snapshots.
    attendance, snapshots_used = mark_attendance_with_voting(
        folder_path, tolerance=tolerance, roll_numbers=roll_numbers
    )

    # Second pass: draw annotated images per-snapshot (unchanged visual
    # behaviour) using the same per-image assignment logic.
    folder = Path(folder_path)
    image_files = sorted(
        [img for img in folder.glob("*") if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
         and not img.name.startswith("annotated_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:5]

    for img_file in image_files:
        print("Processing:", img_file.name)
        encodings, face_locations_hog = detect_faces_in_image(str(img_file))
        # Re-detect at the scale used for annotation (matches previous behaviour:
        # cnn model on a downscaled image, then coords scaled back up)
        image = face_recognition.load_image_file(str(img_file))
        small_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_img, model="cnn")
        encodings = face_recognition.face_encodings(small_img, face_locations)

        draw_img = cv2.imread(str(img_file))
        if draw_img is None:
            pil_img = Image.open(str(img_file)).convert('RGB')
            draw_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if draw_img is None:
            continue

        scale = 2
        assignment = assign_faces_to_students(encodings, face_locations, student_encodings, tolerance)

        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            top, right, bottom, left = top * scale, right * scale, bottom * scale, left * scale
            matched_roll = assignment.get(i)
            color = (0, 255, 0) if matched_roll else (0, 0, 255)
            label = matched_roll if matched_roll else "Unknown"
            cv2.rectangle(draw_img, (left, top), (right, bottom), color, 2)
            cv2.rectangle(draw_img, (left, top - 25), (right, top), color, -1)
            cv2.putText(draw_img, label, (left + 4, top - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        annotated_path = folder / f"annotated_{img_file.name}"
        success = cv2.imwrite(str(annotated_path), draw_img)
        if not success:
            try:
                rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(str(annotated_path))
            except Exception as e:
                print(f"[ANNOTATED] FAILED to save: {e}")

    return {"attendance": attendance, "snapshots_used": snapshots_used}


def mark_attendance_multi_snapshot(class_id, folder_path, session_id, tolerance=0.6, num_snapshots=5):
    print("STEP 1: Loading encodings from DB")

    student_encodings = load_all_student_encodings()
    if not student_encodings:
        print(" No encodings found")
        return {"attendance": {}}

    rolls = list(student_encodings.keys())
    student_vecs = np.array(list(student_encodings.values()))

    # Initialize all students as absent
    final_attendance = {roll: "absent" for roll in rolls}

    folder = Path(folder_path)
    image_files = sorted(
        [img for img in folder.glob("*.jpg")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:num_snapshots]

    if not image_files:
        print("❌ No classroom images found")
        return {"attendance": final_attendance}

    # Process multiple snapshots
    for img_file in image_files:
        print("Processing:", img_file.name)
        image = face_recognition.load_image_file(str(img_file))
        small_img = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

        face_locations = face_recognition.face_locations(small_img, model="cnn")
        encodings = face_recognition.face_encodings(small_img, face_locations)

        print(f"[DEBUG] Faces detected: {len(encodings)}")

        for face_encoding in encodings:
            distances = face_recognition.face_distance(student_vecs, face_encoding)

            for idx, dist in enumerate(distances):
                roll = rolls[idx]
                if dist < tolerance:
                    final_attendance[roll] = "present"
                    print(f" PRESENT: {roll}, Distance: {dist:.4f}")

    print("FINAL:", final_attendance)
    return {"attendance": final_attendance}


def get_student_face_embeddings(roll_number):
    """
    Get face embeddings for a student from their uploaded images
    
    Args:
        roll_number: Student's roll number
    
    Returns:
        list: List of image paths (for compatibility)
    """
    student_folder = STUDENTS_FOLDER / roll_number
    
    if not student_folder.exists():
        return []
    
    # Get all image files from student folder
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
        image_files.extend(student_folder.glob(ext))
    
    return [str(img) for img in image_files]
def resize_image(image, max_width=800):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(image, (0,0), fx=scale, fy=scale)
    return image
    
def get_encodings_upsampled(rgb_img, face_locations, min_face_size=150):
    """
    Upsamples small face crops before encoding to improve accuracy on far faces.
    """
    import cv2
    import face_recognition
    import numpy as np

    encodings = []
    valid_locations = []

    for (top, right, bottom, left) in face_locations:
        # Crop the face
        face_crop = rgb_img[top:bottom, left:right]
        fh, fw = face_crop.shape[:2]

        if fw == 0 or fh == 0:
            continue

        # Upsample if face is smaller than min_face_size
        if fw < min_face_size or fh < min_face_size:
            scale = max(min_face_size / fw, min_face_size / fh)
            face_crop = cv2.resize(
                face_crop,
                (int(fw * scale), int(fh * scale)),
                interpolation=cv2.INTER_CUBIC   # CUBIC > LINEAR for upsampling
            )
            # Recalculate location for the upsampled crop
            enc_location = (0, face_crop.shape[1], face_crop.shape[0], 0)
        else:
            enc_location = (0, fw, fh, 0)

        enc = face_recognition.face_encodings(face_crop, [enc_location])
        if enc:
            encodings.append(enc[0])
            valid_locations.append((top, right, bottom, left))

    return encodings, valid_locations

def mark_attendance_with_voting(folder_path, tolerance: float = None,
                                 min_present_votes: int = MIN_PRESENT_VOTES,
                                 roll_numbers: List[str] = None):
    """
    Run recognition independently on every raw (non-annotated) image in
    folder_path — e.g. the 5-10 snapshots captured across a class session
    every few minutes — and only mark a student Present if they were
    confidently matched in at least `min_present_votes` SEPARATE snapshots.

    This is what actually fixes the "one bad frame = wrong Present" issue:
    a single misidentification in one snapshot can no longer flip someone's
    attendance, because it needs to happen again in another independent
    snapshot to reach the vote threshold.

    Args:
        folder_path: folder containing the session's raw snapshots
        tolerance: base match tolerance (defaults to FACE_MATCH_TOLERANCE)
        min_present_votes: snapshots a student must be matched in
        roll_numbers: optional — restrict matching to this roster only
                      (recommended: pass the class's enrolled roll numbers)

    Returns: (attendance: {roll: 'present'|'absent'}, snapshots_used: int)
    """
    tolerance = tolerance or FACE_MATCH_TOLERANCE
    student_encodings = load_all_student_encodings()
    if roll_numbers is not None:
        student_encodings = {r: v for r, v in student_encodings.items() if r in roll_numbers}

    rolls = list(student_encodings.keys())
    attendance = {roll: "absent" for roll in rolls}
    vote_counts = {roll: 0 for roll in rolls}

    if not student_encodings:
        return attendance, 0

    folder = Path(folder_path)
    image_files = sorted(
        [img for img in folder.glob("*")
         if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
         and not img.name.startswith("annotated_")],
        key=lambda x: x.stat().st_mtime
    )

    snapshots_used = 0
    for img_file in image_files:
        encodings, locations = detect_faces_in_image(str(img_file))
        if not encodings:
            continue
        snapshots_used += 1
        assignment = assign_faces_to_students(encodings, locations, student_encodings, tolerance)
        for matched_roll in set(assignment.values()):  # one vote per student per snapshot
            vote_counts[matched_roll] = vote_counts.get(matched_roll, 0) + 1

    for roll, votes in vote_counts.items():
        if votes >= min_present_votes:
            attendance[roll] = "present"

    print(f"[VOTING] Processed {snapshots_used} snapshot(s). "
          f"Present: {sum(1 for v in attendance.values() if v == 'present')}, "
          f"Absent: {sum(1 for v in attendance.values() if v == 'absent')}")

    return attendance, snapshots_used


def get_dynamic_tolerance(face_location, base_tolerance=0.6):
    top, right, bottom, left = face_location
    face_w = right - left
    face_h = bottom - top
    face_size = min(face_w, face_h)

    if face_size >= 100:
        return base_tolerance          # close face → normal tolerance
    elif face_size >= 60:
        return base_tolerance - 0.05   # medium → 0.55
    elif face_size >= 40:
        return base_tolerance - 0.10   # far → 0.50
    else:
        return base_tolerance - 0.15   # very far → 0.45 (strict, avoid false match)
    
# def assign_faces_to_students(encodings, face_locations, student_vecs, rolls, base_tolerance):
#     if not encodings or student_vecs.size == 0:
#         return {}

#     n_faces = len(encodings)
#     n_students = len(rolls)

#     # Distance matrix: rows = faces, cols = students
#     dist_matrix = np.zeros((n_faces, n_students))
#     tol_per_face = []
#     for i, (enc, loc) in enumerate(zip(encodings, face_locations)):
#         dist_matrix[i] = face_recognition.face_distance(student_vecs, enc)
#         tol_per_face.append(get_dynamic_tolerance(loc, base_tolerance))

#     # Collect every (face, student) pair that individually passes that
#     # face's own tolerance — these are the only candidates worth resolving.
#     candidates = []
#     for i in range(n_faces):
#         tol = tol_per_face[i]
#         for j in range(n_students):
#             if dist_matrix[i, j] < tol:
#                 candidates.append((dist_matrix[i, j], i, j))

#     # Resolve conflicts by taking the globally closest pairs first — this
#     # is what enforces "one face, one student" instead of "whichever
#     # student this face happens to loop over first".
#     candidates.sort(key=lambda c: c[0])

#     used_faces = set()
#     used_students = set()
#     assignment = {}

#     for dist, i, j in candidates:
#         if i in used_faces or j in used_students:
#             continue
#         assignment[i] = rolls[j]
#         used_faces.add(i)
#         used_students.add(j)

#     return assignment 24-07-26

def assign_faces_to_students(encodings, face_locations, student_encodings: Dict[str, List[np.ndarray]],
                              base_tolerance: float):
    """
    One-face-one-student assignment across a single image's detected faces,
    using multi-encoding-per-student corroboration (see match_probe_to_students)
    plus an ambiguity margin between the best and second-best student match,
    plus per-face dynamic tolerance based on face size (far/small faces need
    to clear a stricter bar).

    Returns {face_index: roll_number} for accepted matches only.
    """
    if not encodings or not student_encodings:
        return {}

    n_faces = len(encodings)
    per_face_results = []  # (face_idx, roll, dist, second_dist, tol)

    for i, (enc, loc) in enumerate(zip(encodings, face_locations)):
        tol = get_dynamic_tolerance(loc, base_tolerance)
        roll, dist, second_dist = match_probe_to_students(enc, student_encodings, tol)
        if roll is None:
            continue
        if second_dist is not None and (second_dist - dist) < MIN_CONFIDENCE_MARGIN:
            continue  # too close to a second candidate — ambiguous, reject
        per_face_results.append((i, roll, dist, tol))

    # Resolve one-face-one-student / one-student-one-face conflicts by
    # taking the globally closest matches first.
    per_face_results.sort(key=lambda r: r[2])

    used_faces = set()
    used_students = set()
    assignment = {}
    for face_idx, roll, dist, tol in per_face_results:
        if face_idx in used_faces or roll in used_students:
            continue
        assignment[face_idx] = roll
        used_faces.add(face_idx)
        used_students.add(roll)

    return assignment

# def mark_attendance_fast(class_id, folder_path, session_id, tolerance=0.6):
#     import numpy as np
#     import cv2
#     from pathlib import Path
#     import face_recognition
#     from ultralytics import YOLO

#     # ─── Load YOLOv8 face detection model (download once, cached after) ───────
#     # Using YOLOv8n-face — lightweight, fast, accurate for faces
#     model = YOLO("yolov8m-face-lindevs.pt")  # or full path if you have it locally

#     student_encodings = load_all_student_encodings()
#     if not student_encodings:
#         return {"attendance": {}, "annotated_filename": None}

#     rolls = list(student_encodings.keys())
#     student_vecs = np.array(list(student_encodings.values()))
#     attendance = {roll: "absent" for roll in rolls}
#     folder = Path(folder_path)
#     # ─── Get latest image only ────────────────────────────────────────────────
#     image_files = sorted(
#         [img for img in folder.glob("*")
#          if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
#          and not img.name.startswith("annotated_")],
#         key=lambda x: x.stat().st_mtime,
#         reverse=True
#     )[:1]

#     annotated_filename = None

#     for img_file in image_files:
#         print(f"[INFO] Processing: {img_file.name}")

#         # ─── Load image with OpenCV (BGR) ─────────────────────────────────────
#         draw_img = cv2.imread(str(img_file))
#         if draw_img is None:
#             from PIL import Image as PILImage
#             pil_img = PILImage.open(str(img_file)).convert("RGB")
#             draw_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#             print(f"[ANNOTATE] Used PIL fallback for: {img_file.name}")
#         if draw_img is None:
#             print(f"[ANNOTATE] SKIPPING - could not load image: {img_file}")
#             continue

#         h, w = draw_img.shape[:2]

#         # ─── YOLOv8 face detection ────────────────────────────────────────────
#         # imgsz controls inference resolution — 640 is fast, go up to 1280 for dense crowds
#         results = model.predict(draw_img, imgsz=1280, conf=0.4, verbose=False)
#         boxes = results[0].boxes  # all detections

#         if boxes is None or len(boxes) == 0:
#             print("[INFO] No faces detected by YOLOv8.")
#             # Still save a (un-annotated) copy so the route doesn't break
#             annotated_path = folder / f"annotated_{img_file.name}"
#             cv2.imwrite(str(annotated_path), draw_img)
#             annotated_filename = annotated_path.name
#             continue

#         print(f"[DEBUG] YOLOv8 detected {len(boxes)} face(s)")

#         # ─── Convert YOLO boxes → face_recognition-style crops for encoding ──
#         # face_recognition.face_encodings() needs an RGB numpy array + locations
#         rgb_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)  # face_recognition wants RGB

#         face_locations = []   # (top, right, bottom, left) — face_recognition format
#         for box in boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
#             x1, y1, x2, y2 = map(int, box.tolist())

#             # Clamp to image bounds
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)

#             # face_recognition uses (top, right, bottom, left)
#             face_locations.append((y1, x2, y2, x1))

#         # ─── Get face encodings from the YOLO-detected regions ───────────────
#         encodings, face_locations = get_encodings_upsampled(rgb_img, face_locations, min_face_size=150)
#         print(f"[DEBUG] Encodings generated: {len(encodings)}")

#         # ─── Match each detected face against student DB ────────────────────── 16 July
#         # for face_encoding, face_location in zip(encodings, face_locations):
#         #     top, right, bottom, left = face_location
#         #     tol = get_dynamic_tolerance(face_location, tolerance)
#         #     distances = face_recognition.face_distance(student_vecs, face_encoding)

#         #     matched_roll = None
#         #     best_dist = tol  # only accept if strictly below tolerance

#         #     for idx, dist in enumerate(distances):
#         #         if dist < best_dist:
#         #             best_dist = dist
#         #             matched_roll = rolls[idx]

#         #     if matched_roll:
#         #         attendance[matched_roll] = "present"
#         #         color = (0, 255, 0)
#         #         label = matched_roll
#         #         print(f"PRESENT: {matched_roll}, Distance: {best_dist:.4f}")
#         #     else:
#         #         color = (0, 0, 255)
#         #         label = "Unknown"
#         assignment = assign_faces_to_students(encodings, face_locations, student_vecs, rolls, tolerance)

#         for i, (face_encoding, face_location) in enumerate(zip(encodings, face_locations)):
#             top, right, bottom, left = face_location
#             matched_roll = assignment.get(i)

#             if matched_roll:
#                 attendance[matched_roll] = "present"
#                 color = (0, 255, 0)
#                 label = matched_roll
#                 print(f"PRESENT: {matched_roll}")
#             else:
#                 color = (0, 0, 255)
#                 label = "Unknown"
#             cv2.rectangle(draw_img, (left, top), (right, bottom), color, 2)
#             cv2.rectangle(draw_img, (left, top - 25), (right, top), color, -1)
#             cv2.putText(draw_img, label, (left + 4, top - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#         annotated_path = folder / f"annotated_{img_file.name}"
#         success = cv2.imwrite(str(annotated_path), draw_img)
#         if success:
#             print(f"[ANNOTATED] Saved: {annotated_path.name}")
#             annotated_filename = annotated_path.name
#         else:
#             try:
#                 from PIL import Image as PILImage
#                 rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
#                 PILImage.fromarray(rgb).save(str(annotated_path))
#                 annotated_filename = annotated_path.name
#                 print(f"[ANNOTATED] Saved via PIL fallback: {annotated_path.name}")
#             except Exception as e:
#                 print(f"[ANNOTATED] FAILED to save: {e}")

#     print("[FINAL ATTENDANCE]", attendance)
#     return {
#         "attendance": attendance,
#         "annotated_filename": annotated_filename
#     } 24-7-26

def mark_attendance_fast(class_id, folder_path, session_id, tolerance=None, roll_numbers: List[str] = None):
    """
    Kept for the instant-attendance button, but now uses multi-encoding
    matching (assign_faces_to_students) instead of the old single-vector
    nearest-neighbor loop, so it benefits from the same corroboration fix.
    Still single/few-shot by nature — see the note in chat about optionally
    capturing 2-3 quick snapshots here too for a vote instead of one.
    """
    tolerance = tolerance or FACE_MATCH_TOLERANCE
    from ultralytics import YOLO
    model = YOLO("yolov8m-face-lindevs.pt")

    student_encodings = load_all_student_encodings()
    if roll_numbers is not None:
        student_encodings = {r: v for r, v in student_encodings.items() if r in roll_numbers}
    rolls = list(student_encodings.keys())
    attendance = {roll: "absent" for roll in rolls}

    if not student_encodings:
        return {"attendance": {}, "annotated_filename": None}

    folder = Path(folder_path)
    image_files = sorted(
        [img for img in folder.glob("*")
         if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
         and not img.name.startswith("annotated_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:1]

    annotated_filename = None

    for img_file in image_files:
        draw_img = cv2.imread(str(img_file))
        if draw_img is None:
            pil_img = Image.open(str(img_file)).convert("RGB")
            draw_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if draw_img is None:
            continue

        h, w = draw_img.shape[:2]
        results = model.predict(draw_img, imgsz=1280, conf=0.4, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            annotated_path = folder / f"annotated_{img_file.name}"
            cv2.imwrite(str(annotated_path), draw_img)
            annotated_filename = annotated_path.name
            continue

        rgb_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        face_locations = []
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_locations.append((y1, x2, y2, x1))

        encodings, face_locations = get_encodings_upsampled(rgb_img, face_locations, min_face_size=150)
        assignment = assign_faces_to_students(encodings, face_locations, student_encodings, tolerance)

        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            matched_roll = assignment.get(i)
            if matched_roll:
                attendance[matched_roll] = "present"
                color = (0, 255, 0)
                label = matched_roll
            else:
                color = (0, 0, 255)
                label = "Unknown"
            cv2.rectangle(draw_img, (left, top), (right, bottom), color, 2)
            cv2.rectangle(draw_img, (left, top - 25), (right, top), color, -1)
            cv2.putText(draw_img, label, (left + 4, top - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        annotated_path = folder / f"annotated_{img_file.name}"
        success = cv2.imwrite(str(annotated_path), draw_img)
        if success:
            annotated_filename = annotated_path.name
        else:
            try:
                rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(str(annotated_path))
                annotated_filename = annotated_path.name
            except Exception as e:
                print(f"[ANNOTATED] FAILED to save: {e}")

    return {"attendance": attendance, "annotated_filename": annotated_filename}

def validate_student_images(roll_number):
    """
    Validate that student has uploaded sufficient face images (4-5)
    and that faces can be detected in them
    
    Args:
        roll_number: Student's roll number
    
    Returns:
        bool: True if student has 4-5 images with detectable faces, False otherwise
    """
    if not FACE_RECOGNITION_AVAILABLE:
        student_folder = STUDENTS_FOLDER / roll_number
        if not student_folder.exists():
            return False
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
            image_files.extend(student_folder.glob(ext))
        return 4 <= len(image_files) <= 5
    
    student_folder = STUDENTS_FOLDER / roll_number
    
    if not student_folder.exists():
        return False

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
        image_files.extend(student_folder.glob(ext))
    
    if not (4 <= len(image_files) <= 5):
        return False

    faces_detected = 0
    for image_path in image_files:
        try:
            image = face_recognition.load_image_file(str(image_path))
            face_locations = face_recognition.face_locations(image, model='hog')
            if len(face_locations) > 0:
                faces_detected += 1
        except Exception as e:
            print (f"Error validating face in {image_path} {e}")
            continue
    return faces_detected >= 3

# def process_unregistered_from_image(image_path: str, save_folder: Path, location: str, camera_serial: str) -> Dict[str, int]:
#     if not FACE_RECOGNITION_AVAILABLE:
#         return {'unknown_faces': 0}
#     encodings, locations = detect_faces_in_image(str(image_path))
#     students = load_all_student_encodings()
#     rolls = list(students.keys())
#     student_vecs = np.stack([students[r] for r in rolls]) if rolls else np.empty((0, 128))
#     unknown_locations = []
#     for enc, loc in zip(encodings, locations):
#         if student_vecs.size == 0:
#             unknown_locations.append(loc)
#             continue
#         try:
#             distances = face_recognition.face_distance(student_vecs, enc)
#         except Exception:
#             distances = np.array([
#                 face_recognition.face_distance([students[r]], enc)[0] for r in rolls
#             ])
#         if distances.size == 0:
#             unknown_locations.append(loc)
#             continue
#         order = np.argsort(distances)
#         best_dist = distances[order[0]]
#         second_best = distances[order[1]] if distances.size > 1 else None
#         if best_dist > FACE_MATCH_TOLERANCE:
#             unknown_locations.append(loc)
#             continue
#         if second_best is not None and (second_best - best_dist) < MIN_CONFIDENCE_MARGIN:
#             unknown_locations.append(loc)
#             continue
#         if best_dist > UNKNOWN_FACE_THRESHOLD:
#             unknown_locations.append(loc)
#             continue
#     annotated_path = None
#     if unknown_locations:
#         try:
#             ip = Path(image_path)
#             save_folder.mkdir(parents=True, exist_ok=True)
#             annotated_path = save_folder / f"annotated_{ip.name}"
#             if Image is not None:
#                 im = Image.open(str(ip))
#                 dr = ImageDraw.Draw(im)
#                 for top, right, bottom, left in unknown_locations:
#                     dr.rectangle([(left, top), (right, bottom)], outline=(255, 0, 0), width=3)
#                 im.save(str(annotated_path))
#             elif cv2 is not None:
#                 im = cv2.imread(str(ip))
#                 for top, right, bottom, left in unknown_locations:
#                     cv2.rectangle(im, (left, top), (right, bottom), (0, 0, 255), 3)
#                 cv2.imwrite(str(annotated_path), im)
#         except Exception:
#             annotated_path = None
#     if unknown_locations:
#         try:
#             db.unregistered_detections.insert_one({
#                 'image_path': str(annotated_path or image_path),
#                 'raw_image_path': str(image_path),
#                 'location': location,
#                 'camera_serial': camera_serial,
#                 'timestamp': datetime.now()
#             })
#         except Exception:
#             pass
#     return {'unknown_faces': len(unknown_locations)} 24-07-26

def process_unregistered_from_image(image_path: str, save_folder: Path, location: str, camera_serial: str) -> Dict[str, int]:
    if not FACE_RECOGNITION_AVAILABLE:
        return {'unknown_faces': 0}
    encodings, locations = detect_faces_in_image(str(image_path))
    students = load_all_student_encodings()

    unknown_locations = []
    for enc, loc in zip(encodings, locations):
        roll, dist, second_dist = match_probe_to_students(enc, students, FACE_MATCH_TOLERANCE)
        if roll is None:
            unknown_locations.append(loc)
            continue
        if second_dist is not None and (second_dist - dist) < MIN_CONFIDENCE_MARGIN:
            unknown_locations.append(loc)
            continue
        if dist > UNKNOWN_FACE_THRESHOLD:
            unknown_locations.append(loc)

    annotated_path = None
    if unknown_locations:
        try:
            ip = Path(image_path)
            save_folder.mkdir(parents=True, exist_ok=True)
            annotated_path = save_folder / f"annotated_{ip.name}"
            if Image is not None:
                im = Image.open(str(ip))
                dr = ImageDraw.Draw(im)
                for top, right, bottom, left in unknown_locations:
                    dr.rectangle([(left, top), (right, bottom)], outline=(255, 0, 0), width=3)
                im.save(str(annotated_path))
            elif cv2 is not None:
                im = cv2.imread(str(ip))
                for top, right, bottom, left in unknown_locations:
                    cv2.rectangle(im, (left, top), (right, bottom), (0, 0, 255), 3)
                cv2.imwrite(str(annotated_path), im)
        except Exception:
            annotated_path = None

    if unknown_locations:
        try:
            db.unregistered_detections.insert_one({
                'image_path': str(annotated_path or image_path),
                'raw_image_path': str(image_path),
                'location': location,
                'camera_serial': camera_serial,
                'timestamp': datetime.now()
            })
        except Exception:
            pass

    return {'unknown_faces': len(unknown_locations)}



def mark_attendance_from_images(image_paths):

    known_encodings = []
    known_names = []

    dataset_path = "dataset"

    for person in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person)

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            img = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(img)

            if enc:
                known_encodings.append(enc[0])
                known_names.append(person)

    attendance = {}

    # loop on captured images
    for img_path in image_paths:
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)

        for face in encodings:
            matches = face_recognition.compare_faces(known_encodings, face)

            if True in matches:
                index = matches.index(True)
                name = known_names[index]
                attendance[name] = "present"

    return attendance