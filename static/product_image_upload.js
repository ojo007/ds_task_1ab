document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('productImageForm');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const cnnClassDiv = document.getElementById('cnn-class');
    const responseDiv = document.getElementById('response');
    const productsDiv = document.getElementById('products');

    const API_BASE_URL = 'http://localhost:5000';

    form.addEventListener('submit', async function(e ) {
        e.preventDefault();

        const imageFile = document.getElementById('product_image').files[0];
        if (!imageFile) {
            alert('Please select an image file.');
            return;
        }

        loading.style.display = 'block';
        results.style.display = 'none';

        try {
            const formData = new FormData();
            // --- THIS IS THE FIX ---
            // Key must be 'image' to match app.py request.files['image']
            formData.append('image', imageFile);

            // --- THIS IS THE FIX ---
            // URL must be '/classify' to match the @app.route in app.py
            const response = await fetch(`${API_BASE_URL}/classify`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            loading.style.display = 'none';

            if (response.ok) {
                const identifiedClass = data.predicted_class || 'No class identified';
                cnnClassDiv.innerHTML = `<h3>Identified Product Class:</h3><p><strong>${identifiedClass}</strong></p>`;
                responseDiv.innerHTML = `<h3>Confidence:</h3><p>${(data.confidence * 100).toFixed(2)}%</p>`;
                // You can add logic here to display similar products if your API returns them
                productsDiv.innerHTML = '<p>Related product search can be added here.</p>';
            } else {
                responseDiv.innerHTML = `<h3>Error:</h3><p>${data.error || 'An error occurred'}</p>`;
            }
            results.style.display = 'block';

        } catch (error) {
            loading.style.display = 'none';
            responseDiv.innerHTML = `<h3>Connection Error:</h3><p>Failed to connect to the backend. Is it running?</p><p>${error}</p>`;
            results.style.display = 'block';
        }
    });
});
