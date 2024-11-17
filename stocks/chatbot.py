from langchain.tools import DuckDuckGoSearchRun
from yahooquery import Ticker
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import gradio as gr

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Stock News Tool
class StockNewsTool:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()

    def get_stock_news(self, ticker: str) -> str:
        try:
            query = f"{ticker} stock latest news last 24 hours"
            news = self.search.run(query)
            return news[:1000]  # Truncate to the first 1000 characters
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return f"Unable to fetch recent news for {ticker}. Please try again later."


# Sentiment Analysis Agent
class SentimentAnalysisAgent:
    def __init__(self):
        self.primary_model_name = "tlfmcooper/gemma-2-2b-ft-market-news"
        self.fallback_model_name = "distilbert-base-uncased"  # Replace with any reliable sentiment analysis model
        self.tokenizer = None
        self.model = None
        self.use_fallback = False

        self._initialize_model()

    def _initialize_model(self):
        try:
            # Attempt to load the primary model
            logger.info(f"Loading primary model: {self.primary_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.primary_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.primary_model_name, ignore_mismatched_sizes=True
            )
        except RuntimeError as e:
            # Log the error and switch to fallback
            logger.error(f"Primary model loading failed: {str(e)}")
            logger.info(f"Switching to fallback model: {self.fallback_model_name}")
            self.use_fallback = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.fallback_model_name
            )

    def analyze_sentiment(self, news: str):
        try:
            inputs = self.tokenizer(
                news, return_tensors="pt", truncation=True, padding=True
            )
            outputs = self.model(**inputs)
            probabilities = outputs.logits.softmax(dim=-1)
            sentiment = probabilities.argmax().item()  # 1 for positive, 0 for negative
            return sentiment, probabilities.max().item()
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            return None, None  # Return None values to signify analysis failure


# Recommendation Agent
class RecommendationAgent:
    def provide_recommendation(self, sentiment, probability, news, stock_ticker):
        if sentiment == 1:
            outlook = "positive"
            conclusion = f"Invest in {stock_ticker}."
        else:
            outlook = "negative"
            conclusion = f"Avoid investing in {stock_ticker}."

        # Generate alternative investments dynamically
        alternatives = self.get_alternative_investments(stock_ticker)

        # Markdown-formatted recommendation
        recommendation = f"""
### **Investment Analysis for {stock_ticker.upper()}**

#### **Outlook**: {outlook}  
#### **Probability**: {probability:.2%}  

---

#### **Key Drivers**:  
{news[:500]}...

---

#### **Risk Factors**:  
- Market volatility  
- Geopolitical risks  

---

#### **Alternative Investments**:  
{alternatives}

---

#### **Conclusion**:  
{conclusion}
        """
        return recommendation

    def get_alternative_investments(self, stock_ticker: str):
        try:
            ticker_data = Ticker(stock_ticker)
            sector = ticker_data.summary_profile[stock_ticker]["sector"]
            logger.info(f"Stock {stock_ticker} belongs to sector: {sector}")

            # Fetch peer stocks in the same sector
            sector_peers = ticker_data.get_sector_peers(stock_ticker, sector)
            if not sector_peers:
                return "No suitable alternative investments found."

            # Format peer stocks as Markdown bullet points
            alternatives = [
                f"- **{peer}**"
                for peer in sector_peers
                if peer.upper() != stock_ticker.upper()
            ]
            return (
                "\n".join(alternatives)
                if alternatives
                else "No suitable alternative investments found."
            )
        except Exception as e:
            logger.error(f"Error fetching alternatives for {stock_ticker}: {str(e)}")
            return "Unable to fetch alternative investments at this time."


# Initialize tools and agents
news_tool = StockNewsTool()
sentiment_agent = SentimentAnalysisAgent()
recommendation_agent = RecommendationAgent()


# Orchestration Function
def orchestrate_chatbot(stock_ticker):
    news = news_tool.get_stock_news(stock_ticker)
    sentiment, probability = sentiment_agent.analyze_sentiment(news)
    recommendation = recommendation_agent.provide_recommendation(
        sentiment, probability, news, stock_ticker
    )
    return recommendation


# Gradio Interface
def chatbot_ui(stock_ticker):
    recommendation = orchestrate_chatbot(stock_ticker)
    return recommendation


ui = gr.Interface(
    fn=chatbot_ui,
    inputs=gr.Textbox(label="Enter Stock Ticker"),
    outputs=gr.Markdown(label="Investment Recommendation"),
    title="Stock Market Investment Chatbot",
)

ui.launch()
