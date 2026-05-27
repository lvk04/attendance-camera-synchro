

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
                name = msg["name"]
                if name not in tracker.pending_targets:
                    tracker.pending_targets.append(name)
                    print(f"[TRACK] Target added: {name} — queue size: {len(tracker.pending_targets)}")
                else:
                    print(f"[TRACK] Target skipped (already pending): {name}")
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
        
        # Show multi-target tracking status in overlay
        linked = tracker_app._linked_targets
        pending = tracker_app.pending_targets
        if linked:
            names = list(linked.values())
            label = f"TRACKING: {', '.join(names)}"
            color = (0, 255, 0)
            if pending:
                label += f" (+{len(pending)} waiting)"
        elif pending:
            label = f"WAITING: {len(pending)} target(s) for ROI"
            color = (255, 165, 0)
        else:
            label = "No target set"
            color = (100, 100, 100)
        cv2.putText(small_frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("OSNet ReID Tracking", small_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()