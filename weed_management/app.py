import json
from flask import Flask, request, render_template
import io
from PIL import Image
import base64
import os
from loader import predict_image  # Your model prediction function

app = Flask(__name__)

def image_to_base64(image):
    """Convert image to base64 for display."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Load weed information once at startup
with open("weed_info.json", "r") as json_file:
    weed_info_data = json.load(json_file)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        
        try:
            image = Image.open(io.BytesIO(file.read()))
            predicted_name = predict_image(image).strip()  # Ensure clean formatting
            img_base64 = image_to_base64(image)

            # Retrieve weed info safely
            weed_info = weed_info_data.get(predicted_name, None)

            if weed_info is None:
                weed_info = {
                    "common_name": "Unknown",
                    "scientific_name": "Unknown",
                    "description": "No information available for this weed.",
                    "uses": "N/A"
                }

            return render_template(
                "index.html", 
                img_data=img_base64, 
                predicted_name=predicted_name, 
                weed_info=weed_info
            )

        except Exception as e:
            return render_template("index.html", error=f"Error processing image: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
