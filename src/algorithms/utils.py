# Utility functions for AI job detection and ranking
import re

def is_ai_related(job_title, job_description):
    """Return True if the job is AI-related based on keywords."""
    keywords = [
        'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
        'nlp', 'natural language processing', 'computer vision', 'data science',
        'ai', 'reinforcement learning', 'robotics', 'predictive analytics'
    ]
    text = f"{job_title} {job_description}".lower()
    return any(re.search(rf'\b{kw}\b', text) for kw in keywords)

def rank_jobs(jobs, skill_keywords):
    """Rank jobs by the number of matching skill keywords."""
    def score(job):
        text = f"{job.get('title', '')} {job.get('description', '')}".lower()
        return sum(kw in text for kw in skill_keywords)
    return sorted(jobs, key=score, reverse=True)
