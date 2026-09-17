import os
import numpy as np
import yaml
from src.retrieval.BM25_retriever import BM25Retriever
from src.retrieval.faiss_retriever import get_retriever as get_faiss_retriever

from dotenv import load_dotenv

#Load config
def load_config(config_path = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config   

#Normalize function 
def normalize_scores(scores):
    #1. convert the scores to numpy array
    scores = np.array(scores, dtype=float)

    #2. get the min and max of the scores
    min_s, max_s = scores.min(), scores.max()

    #3. normalize the scores to be between 0 and 1
    if max_s - min_s == 0:
        return np.ones_like(scores) # Avoid division by zero; return an array of ones if all scores are the same

    '''
    scores = [5,5,5]
    min_s = 5
    max_s = 5
    max_s - min_s = 0
    normalized_scores = np.ones_like(scores) -> [1,1,1]
    As a result, all scores are equal and normalized to 1.
    '''

    #4. normalize the scores to be between 0 and 1 using the formula
    return (scores - min_s) / (max_s - min_s)  

#Hybrid retriever class
class HybridRetriever:
    def __init__(self, chunks, config_path = "config.yaml"):
        self.config = load_config(config_path)

        #hybrid settings
        self.k_sparse = self.config["hybrid"]["k_sparse"]
        self.k_semantic = self.config["hybrid"]["k_semantic"]
        self.weights = self.config["hybrid"]["weights"] #weights for the BM25 and FAISS retrievers
        self.normalize = self.config["hybrid"]["normalize"] #whether to normalize the scores or not
        

        self.faiss_retriever = get_faiss_retriever(self.config, chunks_if_needed = chunks)

        self.bm25_retriever = BM25Retriever(chunks, config_path = config_path)

        
    #main retrieve function that combines the results from both retrievers
    def retrieve(self, query, top_k=5):

    # 1. FAISS semantic search
        vectorstore = self.faiss_retriever

        dense_results = vectorstore.similarity_search_with_score(
        query,
        k=self.k_semantic
        )

        dense_raw_scores = [score for doc, score in dense_results]

        # FAISS: lower distance = better
        dense_scores = 1 - normalize_scores(dense_raw_scores)

        # 2. BM25 keyword search
        bm25_results = self.bm25_retriever.get_top_k(
        query,
        k=self.k_sparse
        )

        sparse_raw_scores = [score for doc, score in bm25_results]

        # BM25: higher score = better
        sparse_scores = normalize_scores(sparse_raw_scores)

        # 3. Store scores using document content as key
        combined = {}

         # FAISS results
        for (doc, _), score in zip(dense_results, dense_scores):

            key = doc.page_content.strip()

            combined[key] = {
            "doc": doc,
            "score": self.weights[0] * float(score)
            }

        # BM25 results
        for (doc, _), score in zip(bm25_results, sparse_scores):

            key = doc.page_content.strip()

            if key in combined:

                combined[key]["score"] += (
                self.weights[1] * float(score)
                )

            else:

                combined[key] = {
                "doc": doc,
                "score": self.weights[1] * float(score)
                }

         # 4. Sort by final hybrid score
        ranked_results = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
        )

        # 5. Return top-k
        return [
        (item["doc"], item["score"])
        for item in ranked_results[:top_k]
        ]