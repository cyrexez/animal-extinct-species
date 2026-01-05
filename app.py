from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import io
import requests
from typing import Optional

app = FastAPI(title="Endangered Animal Classifier")

# Load model and class names at startup
try:
    session = ort.InferenceSession("googlenet_animal_classifier.onnx")
    with open('class_names.json', 'r') as f:
        class_mapping = json.load(f)
    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
except Exception as e:
    raise RuntimeError(f"Failed to load model or class names: {e}")

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.HTTPError as e:
        if e.response.status_code == 403:
            raise HTTPException(
                status_code=400, 
                detail="This website blocks automated requests. Please use a direct image URL (ending in .jpg, .png) or try uploading the image instead."
            )
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Extinct Animal Classifier</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3b82f6;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }
        </style>
    </head>
    <body class="bg-white min-h-screen py-12 px-4">
        <div class="max-w-4xl mx-auto">
            <!-- Main Container -->
            <div class="bg-white rounded-3xl shadow-xl border-2 border-blue-100 p-8 md:p-12">
                
                <!-- Header -->
                <div class="text-center mb-8">
                    <h1 class="text-5xl font-bold text-blue-600 mb-3">Animal Classifier</h1>
                    <p class="text-xl text-gray-600">Identify endangered species using AI</p>
                </div>

                <!-- Tabs -->
                <div class="flex border-b-2 border-gray-200 mb-6">
                    <button 
                        onclick="switchTab('upload')" 
                        id="upload-tab-btn"
                        class="tab-btn flex-1 py-4 px-6 text-lg font-semibold text-blue-600 border-b-4 border-blue-600 transition-all"
                    >
                        📁 Upload Image
                    </button>
                    <button 
                        onclick="switchTab('url')" 
                        id="url-tab-btn"
                        class="tab-btn flex-1 py-4 px-6 text-lg font-semibold text-gray-500 border-b-4 border-transparent hover:text-blue-600 transition-all"
                    >
                        🔗 Image URL
                    </button>
                </div>

                <!-- Upload Tab -->
                <div id="upload-tab" class="tab-content">
                    <div 
                        onclick="document.getElementById('fileInput').click()"
                        class="border-4 border-dashed border-blue-400 rounded-2xl p-16 text-center cursor-pointer bg-blue-50 hover:bg-blue-100 hover:border-blue-600 transition-all duration-300 transform hover:-translate-y-1"
                    >
                        <div class="text-7xl mb-4">📸</div>
                        <p class="text-xl font-semibold text-gray-700 mb-2">Click to upload an image</p>
                        <p class="text-sm text-gray-500">Supports JPG, JPEG, PNG</p>
                    </div>
                    <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)" class="hidden">
                </div>

                <!-- URL Tab -->
                <div id="url-tab" class="tab-content hidden">
                    <div class="space-y-4">
                        <input 
                            type="text" 
                            id="urlInput" 
                            placeholder="Enter direct image URL (e.g., https://i.imgur.com/image.jpg)"
                            onkeypress="if(event.key==='Enter') loadImageFromUrl()"
                            class="w-full px-6 py-4 text-lg border-2 border-blue-400 rounded-xl focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-600 transition-all"
                        >
                        
                        <!-- Helpful tips -->
                        <div class="text-sm text-gray-600 bg-blue-50 p-4 rounded-lg border border-blue-200">
                            <p class="font-semibold mb-2">💡 Tips for best results:</p>
                            <ul class="list-disc list-inside space-y-1">
                                <li>Use direct image URLs ending in .jpg, .jpeg, or .png</li>
                                <li>Works well: Imgur, Unsplash, direct image links</li>
                                <li>May not work: iStock, Getty Images, sites requiring login</li>
                            </ul>
                        </div>
                        
                        <button 
                            onclick="loadImageFromUrl()"
                            class="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white py-4 rounded-xl font-bold text-lg hover:from-blue-600 hover:to-blue-700 transform hover:-translate-y-1 transition-all duration-300 shadow-lg hover:shadow-xl"
                        >
                            Load Image
                        </button>
                    </div>
                </div>

                <!-- Preview Image -->
                <div id="preview-container" class="hidden mt-8">
                    <img id="preview" class="max-w-full max-h-96 mx-auto rounded-2xl shadow-2xl border-2 border-blue-100">
                </div>

                <!-- Classify Button -->
                <button 
                    onclick="classifyImage()" 
                    id="classifyBtn"
                    class="hidden w-full mt-8 bg-gradient-to-r from-blue-500 to-blue-600 text-white py-5 rounded-2xl font-bold text-xl hover:from-blue-600 hover:to-blue-700 transform hover:-translate-y-1 transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                    Classify Animal
                </button>

                <!-- Results -->
                <div id="result" class="hidden mt-8 p-8 bg-blue-50 rounded-2xl border-2 border-blue-300"></div>
            </div>
        </div>

        <script>
            let selectedFile = null;
            let currentImageUrl = null;

            function switchTab(tab) {
                // Update tab buttons
                const uploadBtn = document.getElementById('upload-tab-btn');
                const urlBtn = document.getElementById('url-tab-btn');
                
                if (tab === 'upload') {
                    uploadBtn.classList.add('text-blue-600', 'border-blue-600');
                    uploadBtn.classList.remove('text-gray-500', 'border-transparent');
                    urlBtn.classList.add('text-gray-500', 'border-transparent');
                    urlBtn.classList.remove('text-blue-600', 'border-blue-600');
                } else {
                    urlBtn.classList.add('text-blue-600', 'border-blue-600');
                    urlBtn.classList.remove('text-gray-500', 'border-transparent');
                    uploadBtn.classList.add('text-gray-500', 'border-transparent');
                    uploadBtn.classList.remove('text-blue-600', 'border-blue-600');
                }

                // Update tab content
                document.getElementById('upload-tab').classList.toggle('hidden', tab !== 'upload');
                document.getElementById('url-tab').classList.toggle('hidden', tab !== 'url');

                // Reset
                resetState();
            }

            function resetState() {
                selectedFile = null;
                currentImageUrl = null;
                document.getElementById('preview-container').classList.add('hidden');
                document.getElementById('classifyBtn').classList.add('hidden');
                document.getElementById('result').classList.add('hidden');
            }

            function previewImage(event) {
                const file = event.target.files[0];
                if (file) {
                    selectedFile = file;
                    currentImageUrl = null;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        showPreview(e.target.result);
                    }
                    reader.readAsDataURL(file);
                }
            }

            async function loadImageFromUrl() {
                const url = document.getElementById('urlInput').value.trim();
                if (!url) {
                    alert('Please enter an image URL');
                    return;
                }

                currentImageUrl = url;
                selectedFile = null;
                showPreview(url);
            }

            function showPreview(src) {
                document.getElementById('preview').src = src;
                document.getElementById('preview-container').classList.remove('hidden');
                document.getElementById('classifyBtn').classList.remove('hidden');
                document.getElementById('result').classList.add('hidden');
            }

            async function classifyImage() {
                const resultDiv = document.getElementById('result');
                const btn = document.getElementById('classifyBtn');
                
                btn.disabled = true;
                resultDiv.innerHTML = `
                    <div class="flex flex-col items-center">
                        <div class="spinner"></div>
                        <p class="mt-4 text-lg text-gray-700">Analyzing image...</p>
                    </div>
                `;
                resultDiv.classList.remove('hidden');

                try {
                    let response;

                    if (selectedFile) {
                        const formData = new FormData();
                        formData.append('file', selectedFile);
                        response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                    } else if (currentImageUrl) {
                        response = await fetch('/predict-url', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ url: currentImageUrl })
                        });
                    } else {
                        throw new Error('No image selected');
                    }

                    const data = await response.json();

                    if (response.ok) {
                        let html = `
                            <div class="text-center mb-6">
                                <div class="text-4xl font-bold text-blue-600 mb-2">${data.prediction}</div>
                                <div class="text-xl text-blue-500">Confidence: ${data.confidence}</div>
                            </div>
                            <div class="bg-white rounded-xl p-6 shadow-lg border border-blue-100">
                                <h3 class="text-xl font-bold text-blue-600 mb-4">Top 3 Predictions:</h3>
                                <ul class="space-y-3">
                        `;
                        data.top_3.forEach((pred, idx) => {
                            const emoji = idx === 0 ? '🥇' : idx === 1 ? '🥈' : '🥉';
                            const bgColor = idx === 0 ? 'bg-blue-100' : idx === 1 ? 'bg-blue-50' : 'bg-gray-50';
                            const borderColor = idx === 0 ? 'border-blue-500' : idx === 1 ? 'border-blue-400' : 'border-blue-300';
                            html += `
                                <li class="flex items-center justify-between p-4 ${bgColor} rounded-lg border-l-4 ${borderColor}">
                                    <span class="font-semibold text-gray-800">${emoji} ${pred.animal}</span>
                                    <span class="font-bold text-gray-700">${pred.probability}</span>
                                </li>
                            `;
                        });
                        html += `</ul></div>`;
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="text-center text-red-600">
                                <div class="text-4xl mb-2">❌</div>
                                <p class="text-lg font-semibold">Error: ${data.detail}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="text-center text-red-600">
                            <div class="text-4xl mb-2">❌</div>
                            <p class="text-lg font-semibold">Error: ${error.message}</p>
                        </div>
                    `;
                } finally {
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict from uploaded file"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        return await process_prediction(image)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-url")
async def predict_url(data: dict):
    """Predict from image URL"""
    try:
        url = data.get('url')
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        image = load_image_from_url(url)
        return await process_prediction(image)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def process_prediction(image: Image.Image):
    """Common prediction logic"""
    # Preprocess
    img_array = preprocess_image(image)
    
    # Predict
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: img_array})[0]
    probabilities = softmax(logits[0])
    
    # Get results
    predicted_class = np.argmax(probabilities)
    predicted_animal = class_names[predicted_class]
    confidence = probabilities[predicted_class]
    
    # Top 3
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    top_3 = [
        {
            "animal": class_names[idx],
            "probability": f"{probabilities[idx]:.2%}"
        }
        for idx in top_3_idx
    ]
    
    return JSONResponse({
        "prediction": predicted_animal,
        "confidence": f"{confidence:.2%}",
        "top_3": top_3
    })

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}