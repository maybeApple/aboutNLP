"""
Embedding generation and management for semantic search capabilities.
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import hashlib
import re
from config import EMBEDDING_MODEL


class EmbeddingManager:
    """Manages embedding generation and similarity calculations."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model or use fallback."""
        try:
            # Try to use sentence-transformers if available
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to load sentence-transformers: {e}")
            self.logger.info("Using fallback TF-IDF based embeddings")
            self.model = None
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        try:
            if self.model is not None:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                self.logger.info(f"Generated embeddings for {len(texts)} texts")
                return embeddings
            else:
                # Fallback to simple TF-IDF based embeddings
                return self._generate_fallback_embeddings(texts)
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            # Use fallback
            return self._generate_fallback_embeddings(texts)
    
    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate simple TF-IDF based embeddings as fallback."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Simple TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            # Convert to dense array for compatibility
            embeddings = tfidf_matrix.toarray()
            self.logger.info(f"Generated fallback embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            self.logger.error(f"Fallback embedding generation failed: {e}")
            # Return random embeddings as last resort
            return np.random.rand(len(texts), 100)
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_similar_texts(self, query_embedding: np.ndarray, 
                          candidate_embeddings: List[np.ndarray],
                          candidate_texts: List[str],
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar texts based on embedding similarity."""
        similarities = []
        
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate_embedding)
            similarities.append({
                'text': candidate_texts[i],
                'similarity': similarity,
                'index': i
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def batch_similarity_search(self, query_texts: List[str], 
                               candidate_texts: List[str],
                               top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Perform batch similarity search for multiple queries."""
        # Generate embeddings for all texts
        all_texts = query_texts + candidate_texts
        all_embeddings = self.generate_embeddings(all_texts)
        
        # Split embeddings
        query_embeddings = all_embeddings[:len(query_texts)]
        candidate_embeddings = all_embeddings[len(query_texts):]
        
        results = []
        for query_embedding in query_embeddings:
            similarities = self.find_similar_texts(
                query_embedding, 
                candidate_embeddings, 
                candidate_texts, 
                top_k
            )
            results.append(similarities)
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Generate a test embedding to get dimension
        test_embedding = self.generate_single_embedding("test")
        return len(test_embedding)
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str):
        """Save embeddings to a file."""
        np.save(file_path, embeddings)
        self.logger.info(f"Saved embeddings to {file_path}")
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """Load embeddings from a file."""
        embeddings = np.load(file_path)
        self.logger.info(f"Loaded embeddings from {file_path}")
        return embeddings


class SemanticSearchEngine:
    """High-level semantic search engine using embeddings."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.logger = logging.getLogger(__name__)
    
    def build_search_index(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a search index from texts and metadata."""
        if metadata is None:
            metadata = [{'id': i, 'text': text} for i, text in enumerate(texts)]
        
        embeddings = self.embedding_manager.generate_embeddings(texts)
        
        return {
            'embeddings': embeddings,
            'texts': texts,
            'metadata': metadata
        }
    
    def search(self, query: str, search_index: Dict[str, Any], 
               top_k: int = 5, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar texts in the index."""
        query_embedding = self.embedding_manager.generate_single_embedding(query)
        
        similarities = self.embedding_manager.find_similar_texts(
            query_embedding,
            search_index['embeddings'],
            search_index['texts'],
            top_k
        )
        
        # Filter by threshold
        results = []
        for sim in similarities:
            if sim['similarity'] >= similarity_threshold:
                result = {
                    'text': sim['text'],
                    'similarity': sim['similarity'],
                    'metadata': search_index['metadata'][sim['index']]
                }
                results.append(result)
        
        return results
    
    def batch_search(self, queries: List[str], search_index: Dict[str, Any],
                    top_k: int = 5, similarity_threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """Perform batch search for multiple queries."""
        results = []
        for query in queries:
            query_results = self.search(query, search_index, top_k, similarity_threshold)
            results.append(query_results)
        
        return results


if __name__ == "__main__":
    # Test the embedding system
    embedding_manager = EmbeddingManager()
    
    # Test texts
    texts = [
        "Neo4j is a graph database management system",
        "Machine learning algorithms process large datasets",
        "Natural language processing extracts meaning from text",
        "Knowledge graphs represent relationships between entities"
    ]
    
    # Generate embeddings
    embeddings = embedding_manager.generate_embeddings(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Test similarity search
    query = "graph database relationships"
    query_embedding = embedding_manager.generate_single_embedding(query)
    
    similar_texts = embedding_manager.find_similar_texts(
        query_embedding, embeddings, texts, top_k=2
    )
    
    print(f"\nQuery: {query}")
    print("Most similar texts:")
    for result in similar_texts:
        print(f"  {result['similarity']:.3f}: {result['text']}")
