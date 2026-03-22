
# Utility functions for AI job detection and ranking
import re

# --- AI/ML/DS Job Terms Taxonomy (from ai_job_terms_taxonomy.md) ---
TIER1_TITLE_TERMS = [
    # Core Discipline
    'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning', 'dl',
    'data science', 'data scientist', 'natural language processing', 'nlp', 'computer vision', 'cv',
    'generative ai', 'genai', 'large language model', 'llm',
    # Engineering & Operations
    'ml engineer', 'machine learning engineer', 'ai engineer', 'ai developer', 'mlops engineer', 'llmops engineer',
    'data engineer', 'ai/ml platform engineer', 'applied scientist', 'research scientist', 'research engineer',
    # Analytics & Insight
    'data analyst', 'business intelligence', 'bi', 'analytics engineer', 'decision scientist', 'quantitative analyst',
    'quant analyst', 'ai analyst', 'insights analyst',
    # Specialist/Emerging
    'prompt engineer', 'prompt engineering', 'ai product manager', 'ai solutions architect', 'ai strategist',
    'ai consultant', 'conversational ai designer', 'nlp scientist', 'computer vision engineer',
    'recommendation systems engineer', 'ai safety researcher', 'ai ethics specialist', 'ai governance analyst',
    'knowledge graph engineer',
    # Leadership
    'chief ai officer', 'caio', 'head of ai', 'head of data', 'vp of data science', 'director of machine learning',
    'ai practice lead', 'data science manager', 'ml team lead',
]

TIER2_SKILLS_TECH = [
    # Programming & Frameworks
    'python', 'r', 'julia', 'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'sklearn', 'hugging face', 'transformers',
    'langchain', 'llamaindex', 'spacy', 'nltk', 'jax', 'xgboost', 'lightgbm', 'catboost', 'fastapi',
    # Data & Storage
    'sql', 'nosql', 'spark', 'apache spark', 'hadoop', 'kafka', 'dbt', 'pandas', 'numpy', 'polars', 'databricks',
    'snowflake', 'bigquery', 'redshift', 'delta lake', 'feature store', 'vector database', 'pinecone', 'weaviate', 'chroma',
    # Cloud & MLOps
    'aws sagemaker', 'azure machine learning', 'azure ml', 'google vertex ai', 'microsoft fabric', 'kubeflow', 'mlflow',
    'weights & biases', 'w&b', 'dvc', 'airflow', 'apache airflow', 'prefect', 'metaflow', 'bentoml', 'seldon', 'docker',
    'kubernetes', 'ci/cd', 'gitops', 'terraform',
    # LLM & GenAI
    'openai', 'gpt', 'gpt-4', 'gpt-4o', 'claude', 'anthropic', 'gemini', 'google ai', 'llama', 'llama 2', 'llama 3',
    'mistral', 'stable diffusion', 'dall-e', 'copilot', 'bedrock', 'azure openai service', 'model fine-tuning', 'rlhf',
    'rag', 'embeddings', 'vector search', 'semantic search', 'chunking', 'tokenization', 'context window', 'prompt engineering',
]

TIER3_METHODS_CONCEPTS = [
    # ML Concepts
    'supervised learning', 'unsupervised learning', 'reinforcement learning', 'rl', 'transfer learning', 'few-shot learning',
    'zero-shot learning', 'self-supervised learning', 'federated learning', 'neural network', 'transformer', 'attention mechanism',
    'cnn', 'convolutional neural network', 'rnn', 'lstm', 'recurrent neural network', 'gnn', 'graph neural network',
    'diffusion model', 'gan', 'generative adversarial network', 'vae', 'variational autoencoder', 'ensemble methods',
    'gradient boosting', 'random forest', 'hyperparameter tuning', 'cross-validation', 'a/b testing', 'bayesian optimization',
    'automl', 'neural architecture search',
    # NLP
    'sentiment analysis', 'named entity recognition', 'ner', 'text classification', 'information extraction', 'summarization',
    'question answering', 'machine translation', 'text generation', 'speech recognition', 'asr', 'text-to-speech', 'tts',
    'topic modelling', 'lda', 'word embeddings', 'word2vec', 'bert', 'tokenization', 'fine-tuning', 'instruction tuning',
    'peft', 'lora', 'qlora',
    # Computer Vision
    'image classification', 'object detection', 'yolo', 'image segmentation', 'ocr', 'optical character recognition',
    'pose estimation', 'image generation', 'video analysis',
    # Data & Statistics
    'statistical modelling', 'predictive modelling', 'forecasting', 'time series', 'causal inference', 'experiment design',
    'feature engineering', 'feature selection', 'dimensionality reduction', 'pca', 'anomaly detection', 'clustering',
    'recommendation system', 'collaborative filtering', 'data wrangling', 'data cleaning', 'etl', 'elt', 'data governance',
    'data lineage',
    # MLOps & Production
    'model deployment', 'model serving', 'inference', 'model monitoring', 'data drift', 'concept drift', 'model retraining',
    'model versioning', 'pipeline orchestration', 'batch inference', 'real-time inference', 'online inference', 'scalability',
    'distributed computing', 'model evaluation', 'benchmarking', 'responsible ai', 'trustworthy ai', 'ai explainability',
    'xai', 'shap', 'lime', 'ai fairness', 'bias detection', 'ai governance', 'model cards', 'ai safety', 'guardrails', 'red teaming',
]

TIER4_ADJACENT = [
    # Business & Strategy
    'digital transformation', 'ai strategy', 'ai adoption', 'enterprise ai', 'ai implementation', 'data-driven decision making',
    'business intelligence', 'advanced analytics', 'predictive analytics', 'decision intelligence', 'ai roadmap', 'innovation', 'r&d',
    # Infrastructure & Architecture
    'cloud computing', 'cloud native', 'microservices', 'api design', 'rest api', 'graphql', 'data platform', 'data mesh',
    'data lakehouse', 'data warehouse', 'data pipeline', 'real-time data', 'streaming analytics', 'edge computing', 'edge ai',
    # Domain-Specific
    'conversational ai', 'chatbot', 'ai agents', 'agentic ai', 'autonomous systems', 'robotics', 'autonomous vehicles',
    'self-driving', 'drug discovery', 'clinical nlp', 'healthcare ai', 'fintech ai', 'algorithmic trading', 'fraud detection',
    'cybersecurity ai', 'supply chain ai', 'demand forecasting', 'customer segmentation', 'personalization engine',
    'search relevance', 'ranking',
]

# --- Regex Patterns for Heuristic Matching ---
# NOTE: use single backslashes so word boundaries are respected (previous double-escaping
# matched the literal characters "\bAI\b" and missed real tokens).
TITLE_REGEX = re.compile(
    r"machine.?learning|deep.?learning|data.?scien|artificial.?intelligen|\bAI\b|\bML\b|\bNLP\b|\bLLM\b|\bMLOps\b|\bLLMOps\b",
    re.I,
)
SKILLS_REGEX = re.compile(
    r"pytorch|tensorflow|hugging.?face|langchain|scikit|sklearn|llm|generative.?ai|gpt|embedding|rag|fine.?tun|prompt.?engin",
    re.I,
)

def is_ai_related(job_title, job_description):
    """
    Heuristic for AI-related job detection using tiered taxonomy and regex logic.
    Returns True if job is likely AI/ML/DS based on:
      - Strong title match (Tier 1 or regex)
      - Skills/tech match (Tier 2, regex, or >=2 hits)
      - Concepts/methods (Tier 3)
    """
    text = f"{job_title} {job_description}".lower()
    # Strong title match
    if any(term in job_title.lower() for term in TIER1_TITLE_TERMS) or TITLE_REGEX.search(job_title):
        return True
    # Skills/tech match (at least 2 hits)
    skill_hits = sum(term in text for term in TIER2_SKILLS_TECH)
    if skill_hits >= 2 or SKILLS_REGEX.search(text):
        return True
    # Concepts/methods
    if any(term in text for term in TIER3_METHODS_CONCEPTS):
        return True
    return False

def rank_jobs(jobs, skill_keywords):
    """Rank jobs by the number of matching skill keywords across title/description fields."""
    def score(job):
        # Prefer normalized 'jobdescription' if present, otherwise fallback to 'description'
        desc = job.get('jobdescription', job.get('description', ''))
        text = f"{job.get('title', '')} {desc}".lower()
        return sum(kw in text for kw in skill_keywords)
    return sorted(jobs, key=score, reverse=True)
