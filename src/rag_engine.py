import json
from cragmm_search.search import UnifiedSearchPipeline # Internal tool used in Task1.ipynb

class RAGEngine:
    def __init__(self):
        self.search_pipeline = UnifiedSearchPipeline()

    def build_rag_prompt(self, user_query, retrieved_docs, image_description=None):
        """Formats the prompt with external knowledge context."""
        context_str = "\n".join([f"- {doc['text']}" for doc in retrieved_docs[:3]])
        
        prompt = f"""You are a helpful AI assistant. 
        Context from web search:
        {context_str}
        
        Visual Description: {image_description if image_description else "No image provided."}
        
        User Question: {user_query}
        
        Answer the question accurately based on the context and image. 
        If you don't know the answer, say so. Do not hallucinate."""
        
        return prompt

    def get_search_context(self, sub_queries):
        """Aggregates results from multiple sub-queries."""
        all_results = []
        for q in sub_queries:
            results = self.search_pipeline.search(q)
            all_results.extend(results)
        return all_results