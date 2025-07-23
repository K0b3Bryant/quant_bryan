import json
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

class FinancialAnalyzer:
    """
    A class to analyze financial articles using an LLM.
    """
    def __init__(self):
        """
        Initializes the OpenAI client.
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """
        Creates the detailed system prompt for the LLM.
        """
        return """
        You are an expert financial analyst AI specializing in market sentiment. 
        Your task is to analyze a given news article and provide two scores.

        1.  **Fear & Greed Score**: A score on a continuous scale from -100 to 100.
            -   **-100**: Represents extreme FEAR. Associated with words like 'crash', 'panic', 'sell-off', 'recession', 'collapse', 'crisis'.
            -   **0**: Represents NEUTRAL sentiment. The article is balanced, factual, or not strongly opinionated about market direction.
            -   **100**: Represents extreme GREED. Associated with words like 'rally', 'soaring', 'record-high', 'euphoria', 'booming', 'FOMO'.

        2.  **Finance Degree Score**: A score from 0 to 100 indicating how financial the article is.
            -   **0**: The article is not about finance at all (e.g., sports, politics, lifestyle).
            -   **50**: The article mentions finance or economic concepts in passing but is not its main focus (e.g., a political article mentioning the economy).
            -   **100**: The article is a hardcore financial report, market analysis, or earnings report (e.g., a Bloomberg market wrap-up, an SEC filing).

        You MUST respond with a single, valid JSON object and nothing else.
        The JSON object should have exactly two keys: 'fear_greed_score' (integer) and 'finance_degree' (integer).
        
        Example response format:
        {
          "fear_greed_score": -75,
          "finance_degree": 95
        }
        """

    def analyze_article(self, article_text: str) -> dict | None:
        """
        Analyzes the article text and returns the sentiment and finance degree scores.

        Args:
            article_text: The text of the article to analyze.

        Returns:
            A dictionary with the analysis results or None if an error occurs.
        """
        if not article_text:
            print("Warning: Article text is empty. Cannot analyze.")
            return None

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Please analyze the following article text:\n\n{article_text}"}
                ],
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"} # Enforce JSON output
            )
            
            # The API should return a valid JSON string thanks to response_format
            analysis_result = json.loads(response.choices[0].message.content)
            
            # Basic validation
            if 'fear_greed_score' in analysis_result and 'finance_degree' in analysis_result:
                return analysis_result
            else:
                print("Error: LLM response did not contain the required keys.")
                return None

        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from LLM response.")
            return None
        except Exception as e:
            print(f"An error occurred while communicating with the OpenAI API: {e}")
            return None
