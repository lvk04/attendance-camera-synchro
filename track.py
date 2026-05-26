

import cv2
import zmq
import threading

from tracker import PersonTracker


_context = zmq.Context()
_pull = _context.socket(zmq.PULL)
_pull.connect("tcp://127.0.0.1:5557")


def _zmq_listener(tracker):
    while True:
        try:
            msg = _pull.recv_json()
            if msg["action"] == "track":
                tracker.target_name = msg["name"]
                tracker.target_global_id = None
                print(f"[TRACK] Target set: {msg['name']} — waiting for ROI entry")
        except Exception as e:
            print(f"[TRACK] ZMQ error: {e}")


def main():
    # 1. Initialize the tracker (This loads YOLO and OSNet once)
    tracker_app = PersonTracker(
        model_path="yolo11s.pt", 
        reid_weights="osnet_x1_0_msmt17.pth"
         # Or whichever model you downloaded
    )

    threading.Thread(target=_zmq_listener, args=(tracker_app,), daemon=True).start()

    rtsp_url = "rtsp://admin:L26DDDDF@10.40.90.225:554/cam/realmonitor?channel=1&subtype=1"

    # 2. Start the tracking loop
    # The class 'yields' the annotated frame and the list of detection data
    for frame, detections in tracker_app.track(rtsp_url):
        
        # --- FUTURE HOMOGRAPHY STEP ---
        for person in detections:
            track_id = person.get("global_id", person["id"])
            feet_coords = person["base_point"] # (x, y) - use this for mapping!
            
            # Example logic (for later):
            # ground_pt = cv2.perspectiveTransform(np.array([[feet_coords]], dtype='float32'), H)
        # ------------------------------

        # 3. Display Results
        # Resize for performance/viewing
        small_frame = cv2.resize(frame, (480, 384))
        h = small_frame.shape[0]
        
        # Overlay count (detections is a list of dicts)
        unique_ids = set(p.get("global_id", p["id"]) for p in detections)
        count = len(unique_ids)
        cv2.putText(small_frame, f"People: {count}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

        cv2.imshow("OSNet ReID Tracking", small_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()