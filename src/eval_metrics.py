import re
import numpy as np

class Evaluator:
    @staticmethod
    def extract_regex_score(text: str):
        """Extracts scores like 1.0, 0.5, or 0.0 from LLM output."""
        match = re.search(r"(-?1\.0|0\.5|0\.0)", text)
        if match:
            return float(match.group(1))
        return 0.0 # Default fallback if no score found

    @staticmethod
    def calculate_summary_stats(scores_list):
        """Computes final performance metrics for the CRAG-MM benchmark."""
        if not scores_list:
            return 0.0
        return np.mean(scores_list)

    def llm_as_judge(self, model_response, ground_truth):
        """Future implementation for more nuanced semantic scoring."""
        # This would involve another call to a large LLM to compare 
        # semantic similarity between the model response and ground truth.
        pass