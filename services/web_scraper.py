import requests
from bs4 import BeautifulSoup
import os
import time
import pandas as pd
from urllib.parse import urlparse


class WebScraper:
    def __init__(self, base_save_dir="data/scraped_images"):
        self.base_save_dir = base_save_dir
        os.makedirs(self.base_save_dir, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape_images(self, url, product_name, num_images=5):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, "html.parser")
            img_tags = soup.find_all("img")

            saved_image_paths = []
            product_dir = os.path.join(self.base_save_dir, product_name.replace(" ", "_"))
            os.makedirs(product_dir, exist_ok=True)

            for i, img in enumerate(img_tags[:num_images]):
                try:
                    img_url = img.get("src") or img.get("data-src")
                    if not img_url or not img_url.startswith(('http', '//')):
                        continue

                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url

                    img_data = requests.get(img_url, timeout=5).content
                    save_path = os.path.join(product_dir, f"image_{i}.jpg")  # Fixed line

                    with open(save_path, 'wb') as f:
                        f.write(img_data)
                    saved_image_paths.append(save_path)

                except Exception as e:
                    print(f"Error downloading image {i}: {e}")
            return saved_image_paths

        except Exception as e:
            print(f"Scraping error: {e}")
            return []

    def _get_image_url(self, img_tag, domain):
        """Extract image URL based on site structure"""
        img_url = img_tag.get('src') or img_tag.get('data-src')

        if not img_url:
            return None

        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif img_url.startswith('/'):
            img_url = 'https://' + domain + img_url

        return img_url if img_url.startswith('http') else None

    def save_to_csv(self, data, output_path):
        """Save scraped data to CSV"""
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Saved scraped data to {output_path}")


if __name__ == "__main__":
    # Test scraper
    scraper = WebScraper()
    test_url = "https://www.google.com/search?q=white+hanging+heart+t-light+holder&tbm=isch"
    test_product = "WHITE HANGING HEART T-LIGHT HOLDER"
    images = scraper.scrape_images(test_url, test_product, 3)
    print(f"Scraped {len(images)} images")