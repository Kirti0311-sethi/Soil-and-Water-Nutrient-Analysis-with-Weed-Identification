from flask import Flask, render_template, request, redirect
import os
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  # your trained model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#  LANGUAGE DATA
messages = {
    "en": {
        "soil": "Soil Input",
        "water": "Water Input",
        "weed": "Weed Detection",
        "nitrogen": "Nitrogen",
        "ph": "pH",
        "tds": "TDS",
        "submit": "Submit",
        "upload": "Upload Image",
        "soil_bad": "❌ Soil not suitable",
        "water_bad": "❌ Water not safe",
        "weed_detected": "⚠️ Weed Detected",
        "remove": "❌ Remove weeds immediately",
        "clean": "✅ Field is clean",
        "confidence": "Confidence"
    },
    "hi": {
        "soil": "मिट्टी इनपुट",
        "water": "पानी इनपुट",
        "weed": "खरपतवार पहचान",
        "nitrogen": "नाइट्रोजन",
        "ph": "पीएच",
        "tds": "टीडीएस",
        "submit": "जमा करें",
        "upload": "छवि अपलोड करें",
        "soil_bad": "❌ मिट्टी उपयुक्त नहीं",
        "water_bad": "❌ पानी सुरक्षित नहीं",
        "weed_detected": "⚠️ खरपतवार पाया गया",
        "remove": "❌ तुरंत हटाएँ",
        "clean": "✅ खेत साफ है",
        "confidence": "विश्वास स्तर"
    },
    "or": {
        "soil": "ମାଟି ଇନପୁଟ",
        "water": "ପାଣି ଇନପୁଟ",
        "weed": "ଘାସ ଚିହ୍ନଟ",
        "nitrogen": "ନାଇଟ୍ରୋଜେନ",
        "ph": "ପିଏଚ",
        "tds": "ଟିଡିଏସ",
        "submit": "ଦାଖଲ କରନ୍ତୁ",
        "upload": "ଛବି ଅପଲୋଡ୍ କରନ୍ତୁ",
        "soil_bad": "❌ ମାଟି ଭଲ ନୁହେଁ",
        "water_bad": "❌ ପାଣି ସୁରକ୍ଷିତ ନୁହେଁ",
        "weed_detected": "⚠️ ଘାସ ଚିହ୍ନଟ ହେଲା",
        "remove": "❌ ତୁରନ୍ତ ହଟାନ୍ତୁ",
        "clean": "✅ କ୍ଷେତ୍ର ସଫା",
        "confidence": "ବିଶ୍ୱାସ ସ୍ତର"
    },
    "te": {
        "soil": "మట్టి ఇన్‌పుట్",
        "water": "నీటి ఇన్‌పుట్",
        "weed": "కలుపు గుర్తింపు",
        "nitrogen": "నైట్రోజన్",
        "ph": "పీహెచ్",
        "tds": "టిడిఎస్",
        "submit": "సమర్పించండి",
        "upload": "చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "soil_bad": "❌ మట్టి సరైనది కాదు",
        "water_bad": "❌ నీరు సురక్షితం కాదు",
        "weed_detected": "⚠️ కలుపు గుర్తించబడింది",
        "remove": "❌ వెంటనే తొలగించండి",
        "clean": "✅ పొలం శుభ్రంగా ఉంది",
        "confidence": "నమ్మకం స్థాయి"
    }
}

#  HOME
@app.route('/')
def home():
    return render_template("index.html")

#  LANGUAGE PAGE
@app.route('/language')
def language():
    return render_template("language.html")


@app.route('/set_language/<lang>')
def set_language(lang):
    return redirect(f"/soil?lang={lang}")

#  SOIL
@app.route('/soil', methods=['GET', 'POST'])
def soil():
    lang = request.args.get("lang", "en")
    msg = messages[lang]
    error = None

    if request.method == 'POST':
        n = float(request.form.get('n'))
        ph = float(request.form.get('ph'))

        if not (40 <= n <= 80 and 6 <= ph <= 7.5):
            error = msg["soil_bad"]
        else:
            return redirect(f"/water?lang={lang}")

    return render_template("soil.html", msg=msg, lang=lang, error=error)

#  WATER
@app.route('/water', methods=['GET', 'POST'])
def water():
    lang = request.args.get("lang", "en")
    msg = messages[lang]
    error = None

    if request.method == 'POST':
        tds = float(request.form.get('tds'))

        if tds > 1000:
            error = msg["water_bad"]
        else:
            return redirect(f"/weed?lang={lang}")

    return render_template("water.html", msg=msg, lang=lang, error=error)
#  WEED
@app.route('/weed', methods=['GET', 'POST'])
def weed():
    lang = request.args.get("lang", "en")
    msg = messages[lang]

    if request.method == 'POST':
        file = request.files['image']

        # ✅ Save input image
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # 🔥 Run YOLO
        results = model(input_path)

        # ✅ DEFINE output_path BEFORE using it
        output_filename = "result_" + file.filename
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)

        # ✅ Save result image properly
        
        plotted = results[0].plot()
        plotted = cv2.resize(plotted, (500, 350))  # force proper size
        cv2.imwrite(output_path, plotted)

        # ✅ Detection info
        boxes = results[0].boxes
        weed_detected = len(boxes) > 0

        confidence = 0
        if weed_detected:
            confidence = max(boxes.conf).item()

        return render_template("result.html",
                               msg=msg,
                               image_path=output_path,
                               weed_detected=weed_detected,
                               confidence=round(confidence, 2))

    return render_template("weed.html", msg=msg, lang=lang)

if __name__ == "__main__":
    app.run(debug=True)