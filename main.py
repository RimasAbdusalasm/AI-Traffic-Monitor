```python
from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import gc
from datetime import datetime

app = Flask(__name__)

# منع رفع ملفات ضخمة
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# تحميل الموديل مرة واحدة فقط
model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET", "POST"])
def index():

    total = 0
    cars = 0
    trucks = 0
    buses = 0
    motorcycles = 0

    traffic_status = "No Data"
    image_path = None

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                total=0,
                cars=0,
                trucks=0,
                buses=0,
                motorcycles=0,
                status="No Data",
                image=None,
                time=current_time
            )

        file = request.files["image"]

        if file and file.filename != "":

            filepath = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(filepath)

            frame = cv2.imread(filepath)

            if frame is None:
                return render_template(
                    "index.html",
                    total=0,
                    cars=0,
                    trucks=0,
                    buses=0,
                    motorcycles=0,
                    status="Image Error",
                    image=None,
                    time=current_time
                )

            # تقليل حجم الصورة لتقليل استهلاك الذاكرة
            frame = cv2.resize(frame, (640, 640))

            results = model(
                frame,
                imgsz=640,
                conf=0.5,
                verbose=False
            )

            for box in results[0].boxes:

                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "car":
                    cars += 1

                elif label == "truck":
                    trucks += 1

                elif label == "bus":
                    buses += 1

                elif label == "motorcycle":
                    motorcycles += 1

            total = cars + trucks + buses + motorcycles

            if total < 5:
                traffic_status = "Low Traffic"

            elif total < 10:
                traffic_status = "Medium Traffic"

            else:
                traffic_status = "High Traffic"

            annotated = results[0].plot()

            output_path = os.path.join(
                UPLOAD_FOLDER,
                "output.jpg"
            )

            cv2.imwrite(output_path, annotated)

            image_path = "output.jpg"

            # تنظيف الذاكرة
            del frame
            del results
            del annotated
            gc.collect()

    return render_template(
        "index.html",
        total=total,
        cars=cars,
        trucks=trucks,
        buses=buses,
        motorcycles=motorcycles,
        status=traffic_status,
        image=image_path,
        time=current_time
    )

if __name__ == "__main__":
    app.run(debug=True)
```
