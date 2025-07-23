import argparse
from utils.scraper import scrape_article_text
from modules.analyzer import FinancialAnalyzer

def main():
    """
    Main function to run the financial article analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze a financial article for Fear/Greed and Finance Degree.")
    parser.add_argument("url", type=str, help="The URL of the financial article to analyze.")
    args = parser.parse_args()

    print(f"1. Scraping article from: {args.url}")
    article_text = scrape_article_text(args.url)

    if not article_text:
        print("Could not retrieve article content. Exiting.")
        return

    # For debugging, you can print the first 500 chars
    # print(f"   -> Scraped {len(article_text)} characters. Preview: {article_text[:500]}...")

    print("\n2. Analyzing article content with LLM...")
    analyzer = FinancialAnalyzer()
    analysis = analyzer.analyze_article(article_text)

    if not analysis:
        print("Analysis failed. Exiting.")
        return

    print("\n3. Analysis Complete!")
    print("=========================================")
    print(f"Fear/Greed Score: {analysis['fear_greed_score']}")
    print(f"   (-100=Fear, 100=Greed)")
    print(f"Finance Degree: {analysis['finance_degree']}")
    print(f"   (0=Not Financial, 100=Very Financial)")
    print("=========================================")


if __name__ == "__main__":
    main()
