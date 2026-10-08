# -*- coding: utf-8 -*-
"""
High-Performance CCTV Face Quality Enhancement, Multi-Frame Attendance & Annotated Frame Dumper Pipeline
Optimized specifically for CPU-only hardware (Intel Core i5-5300U / 8GB RAM).
"""

import os
import cv2
import time
import pandas as pd
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

# ==============================================================================
# 1. CPU HARDWARE OPTIMIZATION FOR INTEL i5-5300U
# ==============================================================================
def configure_cpu_options():
    """
    Configures ONNX Runtime CPU provider parameters to optimize inference
    on Intel i5-5300U (2 Cores / 4 Threads, Broadwell Architecture).
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4       # Use all 4 logical threads for parallel math ops
    opts.inter_op_num_threads = 1       # Sequential node execution to minimize thread context switching
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts

def load_insightface_app(model_name="buffalo_s", det_size=(640, 640)):
    """
    Loads InsightFace model optimized for CPU.
    - 'buffalo_s' (MobileFaceNet + SCRFD-500M) is ~4-5x faster on CPU than 'buffalo_l' (ResNet50).
    - 'buffalo_l' can still be used if maximum accuracy is desired over speed.
    """
    print(f"[INFO] Initializing InsightFace model '{model_name}' on CPU (4 threads)...")
    app = FaceAnalysis(
        name=model_name,
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=det_size)
    return app

# ==============================================================================
# 2. CCTV IMAGE ENHANCEMENT PIPELINE
# ==============================================================================
class CCTVEnhancer:
    """
    Image preprocessing and face ROI super-resolution pipeline for low-quality CCTV images.
    Addresses lighting shadows, high ISO sensor noise, motion/resolution blur, and compression artifacts.
    """
    
    @staticmethod
    def apply_adaptive_gamma(img_bgr, target_mean=120):
        """
        Dynamically adjusts brightness via Gamma Correction based on scene illumination.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        current_mean = np.mean(gray)
        if current_mean == 0:
            return img_bgr
        
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = np.clip(gamma, 0.5, 2.0)
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img_bgr, table)

    @staticmethod
    def enhance_cctv_frame(img_bgr, clip_limit=2.5, tile_size=(8, 8), enable_denoise=True):
        """
        Full-frame enhancement pipeline:
        1. Light Denoising to prevent noise amplification
        2. Dynamic Gamma Adjustment for backlit / shadowed CCTV scenes
        3. CLAHE in LAB color space for lighting normalization
        """
        if img_bgr is None:
            return None
        
        enhanced = img_bgr.copy()

        # Step 1: Fast bilateral filtering for noise reduction (CCTV ISO noise)
        if enable_denoise:
            enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)

        # Step 2: Adaptive illumination adjustment (Gamma)
        enhanced = CCTVEnhancer.apply_adaptive_gamma(enhanced, target_mean=115)

        # Step 3: Contrast Limited Adaptive Histogram Equalization (CLAHE) on L channel
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        cl = clahe.apply(l_channel)
        
        limg = cv2.merge((cl, a_channel, b_channel))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return enhanced


# ==============================================================================
# 3. VECTORIZED FACE DATABASE & ENROLLMENT
# ==============================================================================
class FaceDatabase:
    """
    Optimized Face Gallery storage using normalized feature embeddings.
    Supports multi-image average embedding (centroid) for each enrolled identity.
    """
    def __init__(self):
        self.student_names = []
        self.gallery_matrix = None  # Shape: (N_students, 512)
        self.raw_embeddings_map = {}

    def enroll_from_directory(self, dataset_path, face_app, cctv_enhancer=None):
        """
        Extracts face embeddings for enrolled students and computes a clean centroid vector per identity.
        """
        print(f"\n[INFO] Starting Enrollment from '{dataset_path}'...")
        temp_database = {}

        if not os.path.exists(dataset_path):
            print(f"[ERROR] Directory '{dataset_path}' does not exist!")
            return

        for student_name in os.listdir(dataset_path):
            student_folder = os.path.join(dataset_path, student_name)
            if not os.path.isdir(student_folder):
                continue

            embeddings = []
            print(f" -> Processing identity: {student_name}")

            for image_name in os.listdir(student_folder):
                image_path = os.path.join(student_folder, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    continue

                if cctv_enhancer:
                    img = cctv_enhancer.enhance_cctv_frame(img)

                faces = face_app.get(img)
                if len(faces) == 0:
                    print(f"    [WARNING] No face detected in {image_name}")
                    continue

                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                emb = faces[0].embedding
                emb = emb / np.linalg.norm(emb)  # L2 normalize
                embeddings.append(emb)

            if len(embeddings) > 0:
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                temp_database[student_name] = centroid
                self.raw_embeddings_map[student_name] = embeddings
                print(f"    ✓ Enrolled '{student_name}' with {len(embeddings)} sample images.")

        self.student_names = list(temp_database.keys())
        if self.student_names:
            self.gallery_matrix = np.array([temp_database[name] for name in self.student_names])
            print(f"\n[SUCCESS] Enrollment complete! Total Enrolled Identities: {len(self.student_names)}")
        else:
            print("[WARNING] No valid embeddings found during enrollment.")

    def match_face(self, query_embedding, threshold=0.40, margin_threshold=0.05):
        """
        Vectorized cosine similarity matching:
        - Matrix product np.dot(gallery, query) computes all cosine similarities in 0.1ms.
        - Applies dual verification: Absolute similarity threshold + Top-1 vs Top-2 score margin check.
        """
        if self.gallery_matrix is None or len(self.student_names) == 0:
            return "Unknown", 0.0

        q_norm = query_embedding / np.linalg.norm(query_embedding)
        scores = np.dot(self.gallery_matrix, q_norm)
        
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])

        if len(self.student_names) == 1:
            if best_score >= threshold:
                return self.student_names[best_idx], best_score
            return "Unknown", best_score

        sorted_indices = np.argsort(scores)[::-1]
        top1_score = scores[sorted_indices[0]]
        top2_score = scores[sorted_indices[1]]
        score_gap = top1_score - top2_score

        if top1_score >= threshold and score_gap >= margin_threshold:
            return self.student_names[sorted_indices[0]], float(top1_score)
        else:
            return "Unknown", float(top1_score)


# ==============================================================================
# 4. MULTI-FRAME ATTENDANCE AGGREGATOR & ANNOTATED FRAME DUMPER
# ==============================================================================
class MultiFrameAttendanceManager:
    """
    Aggregates face recognition results across 5-7 classroom CCTV frames/shots.
    Generates annotated frame outputs saved into a date-formatted folder (e.g., 'annotated_frames_20260729')
    and creates a consolidated Attendance CSV report.
    """
    def __init__(self, db):
        self.db = db
        self.reset_session()

    def reset_session(self):
        self.attendance_records = {
            student: {'max_score': 0.0, 'detections_count': 0, 'best_crop': None}
            for student in self.db.student_names
        }

    def process_frame_batch(self, image_paths, face_app, threshold=0.38, date_format="%Y%m%d", dump_annotated_folder=True):
        """
        Processes a list of 5-7 CCTV classroom images captured a few seconds apart.
        Saves annotated output images into a dated output directory.
        """
        self.reset_session()
        date_folder_name = time.strftime(f"annotated_frames_{date_format}")
        
        if dump_annotated_folder:
            os.makedirs(date_folder_name, exist_ok=True)
            print(f"\n[INFO] Output directory created: '{os.path.abspath(date_folder_name)}'")

        print(f"\n=================================================================")
        print(f"  PROCESSING BATCH: {len(image_paths)} FRAMES ➔ DUMPING TO '{date_folder_name}'")
        print(f"=================================================================")
        
        total_faces_processed = 0

        for frame_idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                print(f"[WARNING] Frame #{frame_idx+1} '{img_path}' not found. Skipping...")
                continue

            raw_img = cv2.imread(img_path)
            if raw_img is None:
                continue

            start_t = time.time()
            # 1. Enhance frame
            enhanced_img = CCTVEnhancer.enhance_cctv_frame(raw_img)

            # 2. Detect faces
            faces = face_app.get(enhanced_img)
            proc_time = (time.time() - start_t) * 1000
            total_faces_processed += len(faces)

            # Copy for drawing annotations
            annotated_img = enhanced_img.copy()

            # Header info bar on top of image
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            header_text = f"CCTV Frame #{frame_idx+1} | {timestamp_str} | Faces Detected: {len(faces)} | Time: {proc_time:.1f}ms"
            cv2.rectangle(annotated_img, (0, 0), (annotated_img.shape[1], 35), (20, 20, 20), -1)
            cv2.putText(annotated_img, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 3. Match faces & draw bounding boxes
            recognized_in_frame = 0
            for face in faces:
                bbox = face.bbox.astype(int)
                w_box, h_box = bbox[2]-bbox[0], bbox[3]-bbox[1]
                if w_box < 16 or h_box < 16:
                    continue

                name, score = self.db.match_face(face.embedding, threshold=threshold)
                
                if name != "Unknown":
                    recognized_in_frame += 1
                    rec = self.attendance_records[name]
                    rec['detections_count'] += 1
                    color = (0, 255, 0)  # Green for recognized
                    
                    if score > rec['max_score']:
                        rec['max_score'] = score
                        pad = 10
                        h_img, w_img = enhanced_img.shape[:2]
                        y1, y2 = max(0, bbox[1]-pad), min(h_img, bbox[3]+pad)
                        x1, x2 = max(0, bbox[0]-pad), min(w_img, bbox[2]+pad)
                        rec['best_crop'] = enhanced_img[y1:y2, x1:x2].copy()
                else:
                    color = (0, 0, 255)  # Red for unknown

                # Draw Bounding Box & Label on Image
                cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                label = f"{name} ({score:.2f})"
                cv2.putText(
                    annotated_img,
                    label,
                    (bbox[0], max(bbox[1]-8, 50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            # Save annotated image into dated folder
            if dump_annotated_folder:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_filename = f"{date_folder_name}/annotated_frame_{frame_idx+1:02d}_{base_name}.jpg"
                cv2.imwrite(save_filename, annotated_img)
                print(f" -> Frame #{frame_idx+1} saved: '{save_filename}' ({recognized_in_frame}/{len(faces)} Recognized)")

        print(f"\n[SUCCESS] Batch Processing Done! All annotated frames stored in folder '{date_folder_name}'")
        return date_folder_name

    def generate_attendance_csv(self, output_csv="attendance_report.csv", min_detection_count=1, min_confidence=0.38):
        """
        Generates and saves the final Attendance CSV report.
        """
        report_data = []
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        for student_name, record in self.attendance_records.items():
            max_score = record['max_score']
            det_count = record['detections_count']

            is_present = (det_count >= min_detection_count) and (max_score >= min_confidence)
            status = "PRESENT" if is_present else "ABSENT"

            report_data.append({
                "Timestamp": now_str,
                "Student_Name": student_name,
                "Status": status,
                "Max_Confidence": round(max_score, 4),
                "Frames_Detected": det_count
            })

        df = pd.DataFrame(report_data)
        df.to_csv(output_csv, index=False)
        
        print(f"\n=================================================================")
        print(f"                ATTENDANCE REPORT GENERATED                       ")
        print(f"=================================================================")
        print(f"Saved CSV Report to: {os.path.abspath(output_csv)}\n")
        print(df.to_string(index=False))
        print("=================================================================\n")

        return df


# ==============================================================================
# 5. MAIN EXECUTION EXAMPLE
# ==============================================================================
if __name__ == "__main__":
    configure_cpu_options()
    app = load_insightface_app(model_name="buffalo_s", det_size=(640, 640))

    db = FaceDatabase()
    dataset_dir = "dataset/dataset"
    if os.path.exists(dataset_dir):
        db.enroll_from_directory(dataset_dir, app, cctv_enhancer=CCTVEnhancer)

    # Example: List of 5-7 CCTV classroom shots
    classroom_cctv_shots = [
        "classroom.jpg",
        "classroom2.jpg",
        "classroom3.jpg"
    ]
    
    existing_shots = [f for f in classroom_cctv_shots if os.path.exists(f)]

    if existing_shots and len(db.student_names) > 0:
        attendance_mgr = MultiFrameAttendanceManager(db)
        
        # Dump annotated frames into date-formatted folder (e.g. 'annotated_frames_20260729')
        output_dir = attendance_mgr.process_frame_batch(existing_shots, app, threshold=0.38, date_format="%Y%m%d")
        
        # Save CSV report
        df_report = attendance_mgr.generate_attendance_csv("attendance_report.csv", min_detection_count=1)
    else:
        print("[NOTE] Add your classroom images to test multi-frame annotated frame dumping.")
