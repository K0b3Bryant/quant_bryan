import requests
from bs4 import BeautifulSoup

def scrape_article_text(url: str) -> str:
    """
    Scrapes the main text content from a given URL.

    Args:
        url: The URL of the article to scrape.

    Returns:
        A string containing the concatenated text of the article's paragraphs.
        Returns an empty string if scraping fails.
    """
    try:
        # Set a user-agent to mimic a browser and avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # A common heuristic: find the main content area and extract paragraphs.
        # This can be improved by targeting specific tags like <article> or divs with "content" ids.
        # For a general approach, we'll just grab all paragraph tags.
        paragraphs = soup.find_all('p')
        
        if not paragraphs:
            # Fallback for pages that don't use <p> tags extensively in the main content
            # This is a very basic fallback and might grab unwanted text.
            return soup.get_text(separator='\n', strip=True)

        article_text = ' '.join([p.get_text() for p in paragraphs])
        
        return article_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during scraping: {e}")
        return ""
