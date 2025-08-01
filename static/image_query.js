document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('imageQueryForm');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const extractedTextDiv = document.getElementById('extracted-text');
    const productsDiv = document.getElementById('products');
    const API_BASE_URL = 'http://localhost:5000';

    form.addEventListener('submit', async function(e ) {
        e.preventDefault();
        const imageFile = document.getElementById('image_data').files[0];
        if (!imageFile) { alert('Please select an image file.'); return; }

        loading.style.display = 'block';
        results.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('image', imageFile); // FIX: Key should be 'image'

            // FIX: Use the correct endpoint '/ocr-query'
            const response = await fetch(`${API_BASE_URL}/ocr-query`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            loading.style.display = 'none';

            if (response.ok) {
                extractedTextDiv.innerHTML = `<h3>Extracted Text:</h3><p><strong>${data.extracted_text || 'N/A'}</strong></p>`;
                if (data.products && data.products.length > 0) {
                    let productsHTML = '<h3>Recommended Products:</h3><table><thead><tr><th>Description</th><th>Price</th><th>Country</th></tr></thead><tbody>';
                    data.products.forEach(product => {
                        productsHTML += `<tr><td>${product.description || 'N/A'}</td><td>${product.unit_price || 'N/A'}</td><td>${product.country || 'N/A'}</td></tr>`;
                    });
                    productsHTML += '</tbody></table>';
                    productsDiv.innerHTML = productsHTML;
                } else {
                    productsDiv.innerHTML = '<p>No products found for the extracted text.</p>';
                }
            } else {
                extractedTextDiv.innerHTML = `<h3>Error:</h3><p>${data.error || 'An unknown error occurred'}</p>`;
            }
            results.style.display = 'block';
        } catch (error) {
            loading.style.display = 'none';
            extractedTextDiv.innerHTML = `<h3>Connection Error:</h3><p>Failed to connect to the backend. Make sure it is running.</p>`;
            results.style.display = 'block';
        }
    });
});
