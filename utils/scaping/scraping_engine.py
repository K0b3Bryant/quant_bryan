import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class ScrapingEngine:
    """Handles the fetching of web page content using different strategies."""

    def __init__(self, headers=None):
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_static(self, url: str) -> str:
        """Fetches a URL using requests. Best for static sites."""
        print(f"Fetching (static): {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching static URL {url}: {e}")
            return ""

    def fetch_dynamic(self, url: str, wait_for_selector: str = None) -> str:
        """Fetches a URL using Selenium. Best for JS-rendered sites."""
        print(f"Fetching (dynamic): {url}")
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless') # Run in background
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(service=service, options=options)
        try:
            driver.get(url)
            if wait_for_selector:
                print(f"Waiting for selector: {wait_for_selector}")
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                )
            # Optional: Give a little extra time for slow-loading elements
            time.sleep(1) 
            return driver.page_source
        except Exception as e:
            print(f"Error fetching dynamic URL {url}: {e}")
            return ""
        finally:
            driver.quit()

    def fetch(self, url: str, strategy: str, wait_for_selector: str = None) -> str:
        """Main fetch method that chooses the strategy."""
        if strategy == "static":
            return self.fetch_static(url)
        elif strategy == "dynamic":
            return self.fetch_dynamic(url, wait_for_selector)
        else:
            raise ValueError(f"Unknown scraping strategy: '{strategy}'")
