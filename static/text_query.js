document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('textQueryForm');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const responseDiv = document.getElementById('response');
    const productsDiv = document.getElementById('products');
    const API_BASE_URL = 'http://localhost:5000';

    form.addEventListener('submit', async function(e ) {
        e.preventDefault();
        const query = document.getElementById('query').value;
        if (!query) { alert('Please enter a search query.'); return; }

        loading.style.display = 'block';
        results.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('query', query);

            // FIX: Use the correct endpoint '/text-search'
            const response = await fetch(`${API_BASE_URL}/text-search`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            loading.style.display = 'none';

            if (response.ok) {
                responseDiv.innerHTML = `<h3>Query:</h3><p>"${data.query}"</p>`;
                if (data.products && data.products.length > 0) {
                    let productsHTML = '<h3>Recommended Products:</h3><table><thead><tr><th>Description</th><th>Price</th><th>Country</th></tr></thead><tbody>';
                    data.products.forEach(product => {
                        productsHTML += `<tr><td>${product.description || 'N/A'}</td><td>${product.unit_price || 'N/A'}</td><td>${product.country || 'N/A'}</td></tr>`;
                    });
                    productsHTML += '</tbody></table>';
                    productsDiv.innerHTML = productsHTML;
                } else {
                    productsDiv.innerHTML = '<p>No products found for this query.</p>';
                }
            } else {
                responseDiv.innerHTML = `<h3>Error:</h3><p>${data.error || 'An unknown error occurred'}</p>`;
            }
            results.style.display = 'block';
        } catch (error) {
            loading.style.display = 'none';
            responseDiv.innerHTML = `<h3>Connection Error:</h3><p>Failed to connect to the backend. Make sure it is running.</p>`;
            results.style.display = 'block';
        }
    });
});
