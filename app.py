import os
import io
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# --- Local Service Imports ---
from services.vector_database import VectorDatabase
from services.ocr_service import OCRService
from services.web_scraper import WebScraper

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- Model Definition (Hardcoded) ---
class PyTorchCNN(nn.Module):
    def __init__(self, num_classes):
        super(PyTorchCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(128 * 16 * 16, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x); x = x.view(x.size(0), -1); x = self.classifier(x); return x

# --- Global Variables & Preprocessing ---
model, class_names, vdb = None, None, None
ocr_service, web_scraper = OCRService(), WebScraper()
preprocess = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def initialize_services():
    global model, class_names, vdb
    print("Initializing services...")
    try:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY", "pcsk_3ipfYv_ACYqDfYGPeAVGixGrZN7SmR8TJQR2zFRriBgRWD3H3zrYCpDeHRnqJkPwzPxpTN")
        vdb = VectorDatabase(api_key=pinecone_api_key, environment="us-east-1")
        print("✅ Vector database initialized successfully.")
    except Exception as e:
        print(f"❌ Vector DB initialization failed: {e}")
    try:
        model_path = os.path.join("models", "pytorch_product_cnn.pth")
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            return
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        num_classes = len(checkpoint['class_to_idx'])
        model = PyTorchCNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        class_names = list(checkpoint['idx_to_class'].values())
        print(f"✅ PyTorch CNN model loaded successfully. Classes: {num_classes}")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")

# --- HTML Page Rendering Routes ---
@app.route("/")
def home(): return render_template("index.html")
@app.route("/text-query-page")
def text_query_page(): return render_template("text_query.html")
@app.route("/image-query-page")
def image_query_page(): return render_template("image_query.html")
@app.route("/product-image-upload-page")
def product_image_upload_page(): return render_template("product_image_upload.html")

# --- API Endpoints ---

@app.route("/classify", methods=["POST"])
def classify_image_endpoint():
    if 'image' not in request.files: return jsonify({"error": "No image file provided."}), 400
    try:
        img = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
        predicted_class = class_names[pred_idx.item()]
        confidence_score = confidence.item()
        products = vdb.search_similar_products(predicted_class, top_k=5) if vdb else []
        return jsonify({"status": "success", "predicted_class": predicted_class, "confidence": float(confidence_score), "products": products})
    except Exception as e:
        return jsonify({"error": "Failed to process image.", "details": str(e)}), 500

@app.route("/text-search", methods=["POST"])
def text_search_endpoint():
    query = request.form.get('query')
    if not query: return jsonify({"error": "No query text provided."}), 400
    if not vdb: return jsonify({"error": "Vector database is not available."}), 503
    try:
        products = vdb.search_similar_products(query, top_k=10)
        return jsonify({"status": "success", "query": query, "products": products})
    except Exception as e:
        return jsonify({"error": "Failed to perform text search.", "details": str(e)}), 500


@app.route("/ocr-query", methods=["POST"])
def ocr_query_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    if not vdb:
        return jsonify({"error": "Vector database is not available."}), 503
    try:
        image_bytes = request.files['image'].read()

        # Try both handwritten and printed text extraction
        extracted_text = ocr_service.extract_text_from_image(image_bytes, is_handwritten=True)

        # If EasyOCR doesn't find much, try Tesseract for printed text
        if not extracted_text or len(extracted_text.strip()) < 3:
            extracted_text = ocr_service.extract_text_from_image(image_bytes, is_handwritten=False)

        if not extracted_text or len(extracted_text.strip()) < 1:
            return jsonify({
                "status": "success",
                "query": "",
                "extracted_text": "No text could be extracted from the image.",
                "products": []
            })

        # Clean up the extracted text
        extracted_text = extracted_text.strip()

        # Search for products using the extracted text
        products = vdb.search_similar_products(extracted_text, top_k=10)

        return jsonify({
            "status": "success",
            "query": extracted_text,
            "extracted_text": extracted_text,
            "products": products
        })

    except Exception as e:
        print(f"OCR query error: {e}")  # Add logging for debugging
        return jsonify({
            "error": "Failed to perform OCR query.",
            "details": str(e)
        }), 500

# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    initialize_services()
    print("\nStarting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
