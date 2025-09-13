from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pyttsx3
from ultralytics import YOLO
import threading
import uvicorn
import time
from queue import Queue, Empty
from threading import Lock

app = FastAPI()

# CORS for frontend (Next.js on 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLO("yolov8s.pt")

# TTS engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# Background TTS queue and worker
tts_queue: Queue[str] = Queue()
tts_running = False

def tts_worker():
    global tts_running
    while tts_running:
        try:
            phrase = tts_queue.get(timeout=0.2)
        except Empty:
            continue
        try:
            engine.say(phrase)
            engine.runAndWait()
        except Exception:
            pass

running = False
latest_jpeg: bytes | None = None
latest_lock: Lock = Lock()
gui_available = True

def run_camera():
    global running
    cap = cv2.VideoCapture(0)
    # Use a reasonable resolution and FPS if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_announce_time = 0.0
    announce_cooldown_seconds = 3.0
    last_announced: set[str] = set()

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Inference
        results = model(frame)

        # Draw detections and collect labels
        current_detected: set[str] = set()
        if results and len(results) > 0:
            r = results[0]
            names = r.names if hasattr(r, "names") else model.names
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    current_detected.add(label)
                    xyxy = box.xyxy[0].int().tolist()
                    x1, y1, x2, y2 = xyxy
                    conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

        # Throttled TTS announcements for new or current objects
        now = time.time()
        should_announce = (now - last_announce_time) >= announce_cooldown_seconds
        new_objects = current_detected - last_announced
        if should_announce and (current_detected or new_objects):
            phrases = []
            # Prioritize new objects; if none, repeat current set occasionally
            announce_set = new_objects if new_objects else current_detected
            for obj in sorted(announce_set):
                phrases.append(f"{obj} ahead")
            for phrase in phrases:
                tts_queue.put(phrase)
            last_announced = set(current_detected)
            last_announce_time = now

        # Publish latest JPEG for MJPEG streaming
        try:
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with latest_lock:
                    # store raw bytes
                    globals()['latest_jpeg'] = buf.tobytes()
        except Exception:
            pass

        # Try native window if GUI available
        if gui_available:
            try:
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break
            except Exception:
                # Disable GUI path if not supported in this environment
                globals()['gui_available'] = False

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

def mjpeg_generator():
    boundary = b'--frame\r\n'
    headers = b'Content-Type: image/jpeg\r\n\r\n'
    while running:
        frame_bytes = None
        with latest_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is not None:
            yield boundary + headers + frame_bytes + b"\r\n"
        time.sleep(0.03)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/view")
def view_page():
    html = (
        """
        <html>
        <head><title>Camera</title></head>
        <body style=\"margin:0; background:#000; display:flex; align-items:center; justify-content:center; height:100vh;\">\n
        <img src=\"/video\" style=\"max-width:100%; max-height:100%;\" />\n
        </body></html>
        """
    )
    return HTMLResponse(content=html)

@app.get("/video")
def video_feed():
    if not running:
        return JSONResponse(content={"message": "Camera not running. Call /start first."}, status_code=400)
    return StreamingResponse(mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/start")
def start_detection():
    global running, tts_running
    if not running:
        running = True
        # Start TTS worker if needed
        if not tts_running:
            tts_running = True
            threading.Thread(target=tts_worker, daemon=True).start()
        # Start camera thread
        threading.Thread(target=run_camera, daemon=True).start()
        return JSONResponse(content={"message": "Camera started"})
    return JSONResponse(content={"message": "Already running"})

@app.get("/stop")
def stop_detection():
    global running, tts_running
    running = False
    tts_running = False
    # Stop any ongoing speech immediately and clear the queue
    try:
        while not tts_queue.empty():
            tts_queue.get_nowait()
    except Exception:
        pass
    engine.stop()
   
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

