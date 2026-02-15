document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const scanningOverlay = document.getElementById('scanningOverlay');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsSection = document.getElementById('resultsSection');
    const errorMsg = document.getElementById('errorMessage');
    const downloadBtn = document.getElementById('downloadReportBtn');

    // UI Elements for Result
    const resStatus = document.getElementById('resStatus');
    const resConfidence = document.getElementById('resConfidence');
    const resClass = document.getElementById('resClass');
    const circleProgress = document.getElementById('circleProgress');

    let uploadedFile = null;

    // 1. Drag & Drop Handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
    dropArea.addEventListener('click', () => fileInput.click()); // Click to upload

    fileInput.addEventListener('change', handleFiles);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files: files } });
    }

    function handleFiles(e) {
        const files = e.target.files;
        if (files.length > 0) {
            uploadedFile = files[0];
            validateAndPreview(uploadedFile);
        }
    }

    function validateAndPreview(file) {
        errorMsg.style.display = 'none';
        resultsSection.style.display = 'none';

        // Basic Type Check
        if (!file.type.startsWith('image/')) {
            showError("Invalid file type. Please upload an MRI or CT Scan image (JPG, PNG).");
            uploadedFile = null;
            return;
        }

        // Preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function () {
            imagePreview.src = reader.result;
            imagePreviewContainer.style.display = 'block';
            analyzeBtn.disabled = false;
        }
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.style.display = 'block';
    }

    // 2. Analyze Button Click
    analyzeBtn.addEventListener('click', async () => {
        if (!uploadedFile) {
            showError("Please upload an image first.");
            return;
        }

        // Check Form Details (Optional but good for report)
        const name = document.getElementById('pName').value;
        if (!name) {
            showError("Please enter Patient Name.");
            return;
        }

        // Start Animation
        scanningOverlay.classList.add('active'); // CSS fade in
        errorMsg.style.display = 'none';
        resultsSection.style.display = 'none';

        // Simulate scanning time if response is too fast (for effect)
        const minTimeArr = new Promise(resolve => setTimeout(resolve, 2000));

        const formData = new FormData();
        formData.append('file', uploadedFile);

        try {
            const [response, _] = await Promise.all([
                fetch('/predict', { method: 'POST', body: formData }),
                minTimeArr
            ]);

            const data = await response.json();

            scanningOverlay.classList.remove('active'); // Hide animation

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || "An unknown error occurred.");
            }

        } catch (error) {
            scanningOverlay.classList.remove('active');
            showError("Server Error: " + error.message);
        }
    });

    function displayResults(data) {
        document.getElementById('initialState').style.display = 'none';
        resultsSection.style.display = 'flex';

        // 1. Text Details
        const isTumor = data.predicted_class.toLowerCase() !== 'no tumor' && data.status !== 'safe'; // Adjust based on your model's labels
        // However, looking at app.py: "status" is reliable ("accept", "reject").
        // Wait, app.py says:
        // "status": "accept" (or "review")
        // "predicted_class": "meningioma" (etc)

        // Let's rely on 'status' and 'predicted_class'
        const tumorLabel = data.predicted_class;
        const confidence = (data.confidence * 100).toFixed(1);

        resClass.textContent = tumorLabel.toUpperCase();
        resConfidence.textContent = confidence + '%';

        // 2. Color Coding & Status
        const tumorLabelLower = tumorLabel.toLowerCase();
        // Check for 'notumor' specifically as defined in config.py
        if (tumorLabelLower === 'notumor' || tumorLabelLower.includes('no tumor') || tumorLabelLower.includes('normal')) {
            resStatus.textContent = "NO TUMOR DETECTED";
            resStatus.className = "result-status status-safe";
            circleProgress.style.stroke = "var(--success-color)";
            resClass.textContent = "NO TUMOR"; // Format nicely
        } else {
            resStatus.textContent = "TUMOR DETECTED";
            resStatus.className = "result-status status-danger";
            circleProgress.style.stroke = "var(--accent-color)";
        }

        // 3. Animate Confidence Circle
        // invalid inputs might result in small confidence if rejected, but here success=true.
        const radius = 70;
        const circumference = 2 * Math.PI * radius; // approx 440
        const offset = circumference - (data.confidence * circumference);

        // Reset first to animate
        circleProgress.style.strokeDashoffset = circumference;
        setTimeout(() => {
            circleProgress.style.strokeDashoffset = offset;
        }, 100);

        // 4. Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // 3. Download Report
    downloadBtn.addEventListener('click', () => {
        const element = document.createElement('div');
        element.innerHTML = `
            <div style="padding: 20px; font-family: sans-serif;">
                <h1 style="color: #0066cc;">MRI Analysis Report</h1>
                <hr>
                <h3>Patient Details</h3>
                <p><strong>Name:</strong> ${document.getElementById('pName').value}</p>
                <p><strong>Age:</strong> ${document.getElementById('pAge').value}</p>
                <p><strong>Gender:</strong> ${document.getElementById('pGender').value}</p>
                <p><strong>Date:</strong> ${new Date().toLocaleDateString()}</p>
                <hr>
                <h3>Scan Analysis</h3>
                <p><strong>Result:</strong> ${resStatus.textContent}</p>
                <p><strong>Classification:</strong> ${resClass.textContent}</p>
                <p><strong>Confidence:</strong> ${resConfidence.textContent}</p>
                <br>
                <img src="${imagePreview.src}" style="max-width: 300px; border-radius: 10px;">
                <br><br>
                <small>Generated by AI Brain Tumor Detection System.</small>
            </div>
        `;

        const opt = {
            margin: 1,
            filename: 'Medical_Report_' + document.getElementById('pName').value + '.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
        };

        // Use html2pdf lib
        html2pdf().set(opt).from(element).save();
    });

});
