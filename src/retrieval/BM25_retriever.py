import numpy as np
import os
import yaml
import re
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def load_config(config_path = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

#tokenize the text into words and return a list of words
def simple_tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text) #re.findall() finds all the words in the text and returns them as a list
    #ignores stop words such as 'the', 'is', 'in', 'and', etc. and removes !, ., ?, etc. from the words
    return tokens

#creating the BM25 class
class BM25Retriever:
    def __init__(self, document, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.document = document
        self.corpus = [simple_tokenize(doc.page_content) for doc in document] #tokenize the document into words and create a corpus of tokenized documents

        #using the corpus to create the BM25 object
        self.bm25 = BM25Okapi(
            self.corpus,
            k1 = self.config["bm25"]["k1"],
            b = self.config["bm25"]["b"])

    #get the top k documents from the BM25 object based on the query and return them as a list of documents
    def get_top_k(self, query, k=None):
            if k is None:
                k = self.config["hybrid"]["k_sparse"] #k returns the top n documents based on the query, if k is not provided, it uses the default value from the config file

            query_tokens = simple_tokenize(query) #tokenize the query

            """
            It compares:

Query
"insurance deductible"

against every document.

For example:

Document 1 → score 8.5
Document 2 → score 3.2
Document 3 → score 0.7
Document 4 → score 5.1

Higher score generally means more relevant.
            """

            scores = self.bm25.get_scores(query_tokens) #get the scores for the query tokens from the BM25 object 

            top_k_indices = np.argsort(scores)[::-1][:k]
            #np.argsort(scores) -> eg: random values [0.1, 0.8, 0.5, 0.3]
            #np.argsort(scores[::-1]) -> sorts and reverses it [0.8, 0.5, 0.3, 0.1] -> [1,2,3,0]
            #np.argsort(scores)[::-1][:2] -> gets the top 2 indices [1, 2]
            
            result = []

            for idx in top_k_indices:
                docs = self.document[idx] #gets the doc at the index
                #self.document[2] -> gets the 3rd document in the document list
                result.append((docs, float(scores[idx]))) #stores the document and its scores as a tuple in the result list

            return result

"""
1. the load_config function loads the configuration from a YAML file.
2. the simple_tokenize function tokenizes the input text into lowercase words, ignoring punctuation and stop words.
3. the BM25Retriever class initializes with a document and configuration, tokenizes the document into a corpus, and creates a BM25 object for scoring.
4. the get_top_k method retrieves the top k documents based on a query, returning them along with their scores.
5. the code uses the rank_bm25 library for BM25 scoring and numpy for array manipulation.
6. the code is designed to be part of a larger system for document retrieval, likely in a hybrid search setup.
"""

