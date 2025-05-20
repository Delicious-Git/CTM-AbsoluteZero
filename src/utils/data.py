"""
Data utilities for CTM-AbsoluteZero.
"""
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger("ctm-az.data")

class DataManager:
    """Data manager for CTM-AbsoluteZero."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory to store data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save
            filename: Filename to save to
            
        Returns:
            Path to saved file
        """
        path = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        logger.debug(f"Saving JSON data to {path}")
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return path
        except Exception as e:
            logger.error(f"Failed to save JSON data to {path}: {e}")
            raise
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load data from a JSON file.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Loaded data
        """
        path = os.path.join(self.data_dir, filename)
        
        logger.debug(f"Loading JSON data from {path}")
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load JSON data from {path}: {e}")
            raise
    
    def save_pickle(self, data: Any, filename: str) -> str:
        """
        Save data to a pickle file.
        
        Args:
            data: Data to save
            filename: Filename to save to
            
        Returns:
            Path to saved file
        """
        path = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        logger.debug(f"Saving pickle data to {path}")
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            return path
        except Exception as e:
            logger.error(f"Failed to save pickle data to {path}: {e}")
            raise
    
    def load_pickle(self, filename: str) -> Any:
        """
        Load data from a pickle file.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Loaded data
        """
        path = os.path.join(self.data_dir, filename)
        
        logger.debug(f"Loading pickle data from {path}")
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Pickle file not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load pickle data from {path}: {e}")
            raise
    
    def save_numpy(self, data: np.ndarray, filename: str) -> str:
        """
        Save NumPy array to a file.
        
        Args:
            data: NumPy array to save
            filename: Filename to save to
            
        Returns:
            Path to saved file
        """
        path = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        logger.debug(f"Saving NumPy data to {path}")
        try:
            np.save(path, data)
            return path
        except Exception as e:
            logger.error(f"Failed to save NumPy data to {path}: {e}")
            raise
    
    def load_numpy(self, filename: str) -> np.ndarray:
        """
        Load NumPy array from a file.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Loaded NumPy array
        """
        path = os.path.join(self.data_dir, filename)
        
        logger.debug(f"Loading NumPy data from {path}")
        try:
            return np.load(path)
        except FileNotFoundError:
            logger.warning(f"NumPy file not found: {path}")
            return np.array([])
        except Exception as e:
            logger.error(f"Failed to load NumPy data from {path}: {e}")
            raise
    
    def list_files(self, pattern: str = "*") -> List[str]:
        """
        List files in the data directory.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of file paths
        """
        import glob
        return glob.glob(os.path.join(self.data_dir, pattern))
    
    def remove_file(self, filename: str) -> bool:
        """
        Remove a file from the data directory.
        
        Args:
            filename: Filename to remove
            
        Returns:
            True if successful, False otherwise
        """
        path = os.path.join(self.data_dir, filename)
        
        logger.debug(f"Removing file {path}")
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to remove file {path}: {e}")
            return False


class VectorDatabase:
    """Simple vector database for storing and retrieving embeddings."""
    
    def __init__(self, data_dir: str = "./embeddings"):
        """
        Initialize the vector database.
        
        Args:
            data_dir: Directory to store embeddings
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.vectors = {}
        self.metadata = {}
    
    def add(
        self,
        key: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a vector to the database.
        
        Args:
            key: Unique identifier for the vector
            vector: Vector to add
            metadata: Optional metadata associated with the vector
        """
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata
    
    def get(self, key: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Get a vector and its metadata from the database.
        
        Args:
            key: Unique identifier for the vector
            
        Returns:
            Tuple of (vector, metadata)
        """
        vector = self.vectors.get(key)
        metadata = self.metadata.get(key)
        return vector, metadata
    
    def remove(self, key: str) -> None:
        """
        Remove a vector from the database.
        
        Args:
            key: Unique identifier for the vector
        """
        if key in self.vectors:
            del self.vectors[key]
        if key in self.metadata:
            del self.metadata[key]
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in the database.
        
        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (key, similarity, metadata) tuples
        """
        if not self.vectors:
            return []
            
        # Calculate cosine similarity
        similarities = {}
        for key, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity >= threshold:
                similarities[key] = similarity
        
        # Sort by similarity (descending)
        sorted_keys = sorted(similarities.keys(), key=lambda k: similarities[k], reverse=True)
        
        # Return top-k results
        results = []
        for key in sorted_keys[:top_k]:
            results.append((key, similarities[key], self.metadata.get(key, {})))
        
        return results
    
    def save(self) -> None:
        """Save the database to disk."""
        data = {
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
            "metadata": self.metadata
        }
        
        try:
            with open(os.path.join(self.data_dir, "vector_db.json"), 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")
            raise
    
    def load(self) -> None:
        """Load the database from disk."""
        try:
            with open(os.path.join(self.data_dir, "vector_db.json"), 'r') as f:
                data = json.load(f)
                
            self.vectors = {k: np.array(v) for k, v in data["vectors"].items()}
            self.metadata = data["metadata"]
        except FileNotFoundError:
            logger.warning("Vector database file not found")
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            raise
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def count(self) -> int:
        """
        Get the number of vectors in the database.
        
        Returns:
            Number of vectors
        """
        return len(self.vectors)