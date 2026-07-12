const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingArea = document.getElementById('loading');
const resultBox = document.getElementById('result-box');
const predictionText = document.getElementById('prediction-text');
const resetBtn = document.getElementById('reset-btn');

let selectedFile = null;

// Handle Clicking on Upload Area
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Drag & Drop Functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelection(e.target.files[0]);
    }
});

function handleFileSelection(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (JPG/PNG).');
        return;
    }
    selectedFile = file;
    // Show Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.classList.add('hidden');
        previewContainer.classList.remove('hidden');
    }
    reader.readAsDataURL(file);
}

// Handle Inference API Call
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI State: Loading
    previewContainer.classList.add('hidden');
    loadingArea.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // Output Results
        loadingArea.classList.add('hidden');
        resultBox.classList.remove('hidden');

        if (response.ok) {
            predictionText.textContent = result.prediction;
        } else {
            predictionText.textContent = 'Error: ' + result.error;
            predictionText.style.color = '#ef4444'; // Red error text
        }
    } catch (err) {
        loadingArea.classList.add('hidden');
        resultBox.classList.remove('hidden');
        predictionText.textContent = 'Server Unreachable';
        predictionText.style.color = '#ef4444';
        console.error(err);
    }
});

// Reset UI
resetBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    resultBox.classList.add('hidden');
    predictionText.style.color = ''; // reset error color
    uploadArea.classList.remove('hidden');
});
