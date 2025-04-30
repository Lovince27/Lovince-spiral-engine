# File: ai_mixed_best.py
import asyncio
import json
import logging
import threading
from collections import Counter
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import cv2
from textblob import TextBlob
from catboost import CatBoostClassifier
import plotly.express as px
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
import ultralytics  # For YOLOv8
import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AIMixerBest:
    """Advanced AI pipeline for product analysis with LangChain as the head."""
    
    def __init__(self, output_dir: str = "results_best"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.reviews: List[Dict] = []
        self.model = CatBoostClassifier(random_state=42, verbose=False)
        self.llm = self._setup_langchain()
        self.lock = threading.Lock()

    def _setup_langchain(self) -> LLMChain:
        """Set up LangChain with distilgpt2 as the head."""
        logger.info("Setting up LangChain with distilgpt2...")
        hf_pipeline = pipeline("text-generation", model="distilgpt2", max_length=50)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Analyze this review: {text}. Provide sentiment (positive/negative/neutral) and a brief summary."
        )
        return LLMChain(llm=llm, prompt=prompt)

    async def fetch_reviews_async(self, query: str = "Olay product review", limit: int = 10) -> List[Dict]:
        """Fetch reviews asynchronously (mocked for demo; replace with real API)."""
        logger.info(f"Fetching reviews for query: {query}")
        async with aiohttp.ClientSession() as session:
            # Mock API call (replace with real X/web scraping API)
            @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
            async def mock_api():
                return [
                    {
                        "text": f"Olay {random.choice(['cream', 'serum'])} is {'great' if i % 2 else 'okay'}!",
                        "source": "mock"
                    }
                    for i in range(limit)
                ]
            reviews = await mock_api()
            self.reviews = [
                {
                    "text": r["text"],
                    "length": len(r["text"].split()),
                    "satisfaction": 5 if "great" in r["text"] else 3  # Dummy label
                }
                for r in reviews
            ]
            return self.reviews

    def analyze_sentiment(self, text: str) -> Tuple[str, float, str]:
        """Analyze sentiment with TextBlob and cross-check with LangChain."""
        logger.info(f"Analyzing sentiment for text: {text[:20]}...")
        # TextBlob
        blob = TextBlob(text)
        tb_polarity = blob.sentiment.polarity
        tb_label = "positive" if tb_polarity > 0 else "negative" if tb_polarity < 0 else "neutral"
        
        # LangChain
        try:
            lc_result = self.llm.run(text)
            lc_label = "positive" if "positive" in lc_result.lower() else "negative" if "negative" in lc_result.lower() else "neutral"
        except Exception as e:
            logger.warning(f"LangChain failed: {e}. Using TextBlob.")
            lc_label = tb_label
        
        # Cross-check
        if tb_label != lc_label:
            logger.warning(f"Sentiment mismatch: TextBlob={tb_label}, LangChain={lc_label}")
            final_label = tb_label  # Default to TextBlob if mismatch
        else:
            final_label = tb_label
        
        return final_label, tb_polarity, lc_result

    def extract_topics(self, texts: List[str], top_n: int = 5) -> List[str]:
        """Extract topics using Counter (core Python)."""
        logger.info("Extracting topics...")
        words = [word.lower() for text in texts for word in text.split() if len(word) > 3 and word.isalpha()]
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_n)]

    def train_ml_model(self) -> None:
        """Train CatBoost classifier on review features."""
        logger.info("Training CatBoost model...")
        if not self.reviews:
            logger.error("No reviews to train on!")
            return
        X = np.array([[r["length"], TextBlob(r["text"]).sentiment.polarity] for r in self.reviews])
        y = np.array([r["satisfaction"] for r in self.reviews])
        self.model.fit(X, y)

    def predict_satisfaction(self, text: str, length: int) -> int:
        """Predict satisfaction score with cross-check."""
        polarity = TextBlob(text).sentiment.polarity
        X_new = np.array([[length, polarity]])
        pred = self.model.predict(X_new)[0]
        # Self-check: Ensure prediction aligns with sentiment
        if polarity > 0 and pred < 3:
            logger.warning(f"Prediction ({pred}) inconsistent with positive sentiment. Adjusting...")
            pred = 3
        return pred

    def process_image(self, image_path: str) -> np.ndarray:
        """Detect objects using YOLOv8."""
        logger.info(f"Processing image: {image_path}")
        try:
            model = ultralytics.YOLO("yolov8n.pt")  # Nano model for speed
            img = cv2.imread(image_path)
            if img is None:
                logger.error("Image not found!")
                return None
            results = model(img)
            return results[0].plot()  # Rendered image with bounding boxes
        except Exception as e:
            logger.error(f"YOLOv8 failed: {e}")
            return None

    def save_results(self, results: List[Dict]) -> None:
        """Save results to JSON for API compatibility."""
        output_file = self.output_dir / "ai_results.json"
        logger.info(f"Saving results to {output_file}")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def visualize_results(self, results: List[Dict]) -> None:
        """Create interactive Plotly visualization."""
        logger.info("Generating interactive visualization...")
        df = {
            "index": range(len(results)),
            "satisfaction": [r["satisfaction"] for r in results],
            "sentiment": [1 if r["sentiment"] == "positive" else 0 for r in results]
        }
        fig = px.scatter(
            df, x="index", y="satisfaction", color="sentiment",
            title="Sentiment vs. Satisfaction Analysis",
            labels={"index": "Review Index", "satisfaction": "Predicted Satisfaction"}
        )
        fig.write_html(self.output_dir / "visualization.html")
        fig.show()

    async def process_reviews_async(self) -> List[Dict]:
        """Process reviews asynchronously with LangChain orchestration."""
        logger.info("Processing reviews asynchronously...")
        results = []

        async def process_review(review: Dict) -> Dict:
            with self.lock:
                sentiment, polarity, lc_summary = await asyncio.get_event_loop().run_in_executor(
                    None, self.analyze_sentiment, review["text"]
                )
                satisfaction = await asyncio.get_event_loop().run_in_executor(
                    None, self.predict_satisfaction, review["text"], review["length"]
                )
                return {
                    "text": review["text"],
                    "sentiment": sentiment,
                    "satisfaction": satisfaction,
                    "summary": lc_summary,
                    "topics": []
                }

        tasks = [process_review(review) for review in self.reviews[:10]]  # Limit for demo
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in results if not isinstance(r, Exception)]
        
        # Add topics
        topics = self.extract_topics([r["text"] for r in self.reviews])
        for r in results:
            r["topics"] = topics
        return results

    async def run(self, image_path: str = "olay_product.jpg") -> None:
        """Main async method to run AI pipeline."""
        logger.info("Starting AI pipeline...")
        await self.fetch_reviews_async()
        self.train_ml_model()
        results = await self.process_reviews_async()
        self.save_results(results)
        self.visualize_results(results)
        
        # Process image
        processed_img = self.process_image(image_path)
        if processed_img is not None:
            cv2.imwrite(str(self.output_dir / "processed_image.jpg"), processed_img)
            plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            plt.title("Detected Objects in Product Image")
            plt.axis("off")
            plt.savefig(self.output_dir / "image_plot.png")
            plt.show()

if __name__ == "__main__":
    mixer = AIMixerBest()
    asyncio.run(mixer.run())