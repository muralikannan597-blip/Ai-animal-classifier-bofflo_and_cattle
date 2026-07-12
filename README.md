# AI Animal Classifier (Buffalo & Cattle) 🐄🦬

An end-to-end Machine Learning web application that classifies uploaded images of animals into either **Buffalo** or **Cattle**.

### 🌐 [Live Demo URL: https://your-app-name.onrender.com]
*(Once your Render deployment is finished, update this link to your actual Render URL!)*

## Overview
This project predicts whether a given image contains cattle or a buffalo. It was built completely from scratch over three major modules:
1. **Data Processing**: Extracting and transforming images for training (Data Augmentation using `torchvision.transforms`).
2. **Model Training**: Utilizing a **ResNet-18** deep learning architecture (via PyTorch) and modifying the final connected layer to classify into our 2 custom target categories.
3. **Web Application**: A complete web deployment featuring a Flask backend API and a modern, full-styled glassmorphic frontend UI.

## Tech Stack
- **Backend / AI Engine**: Python, Flask, PyTorch, Torchvision, Pillow
- **Frontend UI**: HTML5, CSS3, Vanilla JavaScript
- **Deployment Ready**: Fully configured `.gitignore` and `requirements.txt` included for seamless cloud hosting (e.g., Render.com).

## How to Run It Locally

### 1. Start the Backend API (Flask)
Navigate to the `backend/` directory, install the necessary requirements, and run `app.py`:
```bash
cd backend
pip install -r requirements.txt
python app.py
```
*The Flask inference server will boot up and wait for requests on `http://127.0.0.1:5000`.*

### 2. Start the Frontend Application
For the user interface, you can either:
- Simply open `frontend/index.html` directly in your favorite browser.
- Or run a localized server via Python in the `frontend` folder:
```bash
cd frontend
python -m http.server 8080
```
*Then navigate to `http://localhost:8080` in your web browser, upload an image, and test the AI!*

## The Model
The AI engine runs on `livestock_model.pth`. It was fine-tuned for 5 epochs on CPU and exhibits incredible validation accuracy in differentiating between the two species.

---
*Built with ❤️.*
