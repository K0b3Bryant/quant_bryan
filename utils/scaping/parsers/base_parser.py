from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class BaseParser(ABC):
    """Abstract Base Class for all parsers."""
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.url = config['url']

    @abstractmethod
    def parse(self, html: str) -> list:
        """Parses the HTML content to extract structured data."""
        pass

    def scrape(self, max_pages=1):
        """
        Main method to orchestrate scraping, including pagination.
        
        Args:
            max_pages (int): The maximum number of pages to scrape.
        """
        all_results = []
        current_url = self.url
        
        for i in range(max_pages):
            print(f"\n--- Scraping Page {i+1}: {current_url} ---")
            
            html = self.engine.fetch(
                current_url,
                self.config['strategy'],
                self.config['parser_config'].get('item_selector')
            )
            
            if not html:
                print("Failed to fetch HTML. Stopping.")
                break

            soup = BeautifulSoup(html, 'html.parser')
            page_results = self.parse(soup)
            all_results.extend(page_results)
            print(f"Found {len(page_results)} items on this page.")
            
            # --- Pagination Logic ---
            p_config = self.config['parser_config'].get('pagination')
            if not p_config:
                break # No pagination configured

            next_page_element = soup.select_one(p_config['selector'])
            if not next_page_element:
                print("No 'next page' link found. Ending scrape.")
                break
                
            next_page_url = next_page_element.get(p_config.get('attribute', 'href'))
            if not next_page_url:
                print("Next page link found, but no URL. Ending scrape.")
                break
            
            # Create an absolute URL from a relative one
            current_url = urljoin(current_url, next_page_url)
        
        return all_results
