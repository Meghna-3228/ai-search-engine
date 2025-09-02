# ai-search-engine
An AI-powered search &amp; exploration tool. Features neural-symbolic reasoning, multi-level caching, adaptive ranking, and "what-if" queries for a smarter search experience.

Welcome to the AI-Powered Search Engine! This isn't your average search tool. It's a smart, context-aware engine designed to understand what you really mean and give you trustworthy, relevant results.

What's This All About?
Ever feel like you're fighting with a search engine, trying endless combinations of words just to find that one thing you're looking for? This project was built to fix that!

Traditional search engines are good at matching keywords, but they often miss the real meaning and context behind your search. Our goal is to bridge the gap between what you're thinking and the results you get. We're making search more intuitive, personalized, and reliable.

Key Features
This search engine is packed with cool features that make it stand out:

Neural-Symbolic Reasoning: Combines the pattern-recognition power of neural networks with the precision of symbolic logic. This means it can understand complex queries like "tech companies with female CEOs and growing revenue."

Multi-Level Caching: A super-smart caching system (memory, disk, and database) that learns from your search patterns and groups topics together to deliver results faster.

Dynamic Query Processing: Goes beyond keywords to understand your intent, adds context from your past searches, and even expands your query with related terms to find the best results.

Adaptive Ranking (with Reinforcement Learning): Learns from how you interact with results (what you click on, how long you stay on a page) to constantly get better at ranking, personalizing the experience for you.

"What-If" Explorer (Counterfactual Reasoning): Ask hypothetical questions like "What if the internet was never invented?" and get reasoned, exploratory answers. This is something most search engines just can't do!

Result Verification: Helps you understand where the information comes from by analyzing sources for trustworthiness and consistency, fighting back against misinformation.

1. Prerequisites
Make sure you have Python 3.8+ installed on your system. You'll also need a package manager like pip.

2. Installation
Clone the repository to your local machine:

git clone https://github.com/Meghna-3228/ai-search-engine.git
cd ai-search-engine

Install all the necessary Python packages.

pip install -r requirements.txt

3. Configuration
This project needs API keys to connect to different AI services.

Find the .env file in src folder. Open the .env file and add your API keys:

COHERE_API_KEY="your_cohere_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"

4. Running the App
Launch the Streamlit application with this command:

streamlit run app.py

Your browser should open a new tab with the running application. Have fun searching! 🎉

Screenshots
Here’s a sneak peek of what the app looks like in action!

<img width="979" height="442" alt="Picture1" src="https://github.com/user-attachments/assets/5c014ad3-ce88-4074-a087-c7344c66346c" />
