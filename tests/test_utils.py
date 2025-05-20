"""
Tests for utility modules.
"""
import os
import sys
import unittest
import tempfile
import json
import yaml
import numpy as np
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logging import CTMLogger, PerformanceTracker
from src.utils.config import ConfigManager, load_config, merge_configs
from src.utils.data import DataManager, VectorDatabase


class TestLogging(unittest.TestCase):
    """Tests for the logging module."""
    
    def test_ctm_logger(self):
        """Test CTMLogger class."""
        with tempfile.NamedTemporaryFile() as log_file:
            # Create logger with file output
            logger = CTMLogger(
                name="test-logger",
                log_level=logging.INFO,
                log_file=log_file.name,
                console_output=False
            ).get_logger()
            
            # Log a message
            logger.info("Test message")
            
            # Check that the message was written to the file
            with open(log_file.name, 'r') as f:
                log_content = f.read()
                self.assertIn("Test message", log_content)
    
    def test_performance_tracker(self):
        """Test PerformanceTracker class."""
        tracker = PerformanceTracker()
        
        # Test timer
        tracker.start_timer("test_timer")
        tracker.stop_timer("test_timer")
        self.assertIn("time_test_timer", tracker.metrics)
        self.assertEqual(len(tracker.metrics["time_test_timer"]), 1)
        
        # Test counter
        tracker.increment_counter("test_counter")
        tracker.increment_counter("test_counter", 2)
        self.assertEqual(tracker.get_counter("test_counter"), 3)
        self.assertIn("count_test_counter", tracker.metrics)
        self.assertEqual(len(tracker.metrics["count_test_counter"]), 2)
        
        # Test custom metric
        tracker.record_metric("test_metric", 42)
        self.assertIn("test_metric", tracker.metrics)
        self.assertEqual(tracker.metrics["test_metric"][0], 42)
        
        # Test reset
        tracker.reset()
        self.assertEqual(len(tracker.metrics), 0)
        self.assertEqual(len(tracker.timers), 0)
        self.assertEqual(len(tracker.counters), 0)


class TestConfig(unittest.TestCase):
    """Tests for the config module."""
    
    def test_config_manager(self):
        """Test ConfigManager class."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as config_file:
            # Create test config
            test_config = {
                "key1": "value1",
                "key2": 42,
                "nested": {
                    "inner_key": "inner_value"
                }
            }
            
            # Write config to file
            yaml.dump(test_config, config_file)
            config_file.flush()
            
            # Load config
            config_manager = ConfigManager(config_file.name)
            
            # Test basic get
            self.assertEqual(config_manager.get("key1"), "value1")
            self.assertEqual(config_manager.get("key2"), 42)
            
            # Test nested get
            self.assertEqual(config_manager.get("nested.inner_key"), "inner_value")
            
            # Test default value
            self.assertEqual(config_manager.get("non_existent", "default"), "default")
            
            # Test set
            config_manager.set("new_key", "new_value")
            self.assertEqual(config_manager.get("new_key"), "new_value")
            
            # Test nested set
            config_manager.set("nested.new_inner", "new_inner_value")
            self.assertEqual(config_manager.get("nested.new_inner"), "new_inner_value")
            
            # Test to_dict
            config_dict = config_manager.to_dict()
            self.assertEqual(config_dict["key1"], "value1")
            self.assertEqual(config_dict["nested"]["inner_key"], "inner_value")
    
    def test_merge_configs(self):
        """Test merge_configs function."""
        base_config = {
            "key1": "value1",
            "key2": 42,
            "nested": {
                "inner1": "inner_value1",
                "inner2": 100
            }
        }
        
        override_config = {
            "key1": "new_value1",
            "nested": {
                "inner1": "new_inner_value1",
                "inner3": "inner_value3"
            },
            "key3": "value3"
        }
        
        merged = merge_configs(base_config, override_config)
        
        # Test merged values
        self.assertEqual(merged["key1"], "new_value1")  # Overridden
        self.assertEqual(merged["key2"], 42)  # Unchanged
        self.assertEqual(merged["key3"], "value3")  # Added
        self.assertEqual(merged["nested"]["inner1"], "new_inner_value1")  # Nested overridden
        self.assertEqual(merged["nested"]["inner2"], 100)  # Nested unchanged
        self.assertEqual(merged["nested"]["inner3"], "inner_value3")  # Nested added


class TestData(unittest.TestCase):
    """Tests for the data module."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_manager = DataManager(self.test_dir.name)
    
    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()
    
    def test_json_save_load(self):
        """Test JSON save and load."""
        test_data = {"key": "value", "nested": {"inner": 42}}
        
        # Save data
        path = self.data_manager.save_json(test_data, "test.json")
        self.assertTrue(os.path.exists(path))
        
        # Load data
        loaded_data = self.data_manager.load_json("test.json")
        self.assertEqual(loaded_data, test_data)
    
    def test_pickle_save_load(self):
        """Test pickle save and load."""
        test_data = {"key": "value", "array": np.array([1, 2, 3])}
        
        # Save data
        path = self.data_manager.save_pickle(test_data, "test.pkl")
        self.assertTrue(os.path.exists(path))
        
        # Load data
        loaded_data = self.data_manager.load_pickle("test.pkl")
        self.assertEqual(loaded_data["key"], test_data["key"])
        self.assertTrue(np.array_equal(loaded_data["array"], test_data["array"]))
    
    def test_numpy_save_load(self):
        """Test NumPy save and load."""
        test_array = np.array([[1, 2], [3, 4]])
        
        # Save data
        path = self.data_manager.save_numpy(test_array, "test.npy")
        self.assertTrue(os.path.exists(path))
        
        # Load data
        loaded_array = self.data_manager.load_numpy("test.npy")
        self.assertTrue(np.array_equal(loaded_array, test_array))
    
    def test_list_files(self):
        """Test listing files."""
        # Create test files
        self.data_manager.save_json({"key": "value"}, "test1.json")
        self.data_manager.save_json({"key": "value"}, "test2.json")
        
        # List files
        files = self.data_manager.list_files("*.json")
        self.assertEqual(len(files), 2)
    
    def test_remove_file(self):
        """Test removing files."""
        # Create test file
        path = self.data_manager.save_json({"key": "value"}, "test.json")
        self.assertTrue(os.path.exists(path))
        
        # Remove file
        result = self.data_manager.remove_file("test.json")
        self.assertTrue(result)
        self.assertFalse(os.path.exists(path))


class TestVectorDatabase(unittest.TestCase):
    """Tests for the VectorDatabase class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.vector_db = VectorDatabase(self.test_dir.name)
    
    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()
    
    def test_add_get(self):
        """Test adding and getting vectors."""
        vector = np.array([1.0, 2.0, 3.0])
        metadata = {"key": "value"}
        
        # Add vector
        self.vector_db.add("test1", vector, metadata)
        
        # Get vector
        retrieved_vector, retrieved_metadata = self.vector_db.get("test1")
        
        # Check results
        self.assertTrue(np.array_equal(retrieved_vector, vector))
        self.assertEqual(retrieved_metadata, metadata)
    
    def test_search(self):
        """Test vector search."""
        # Add test vectors
        self.vector_db.add(
            "test1",
            np.array([1.0, 0.0, 0.0]),
            {"name": "test1"}
        )
        self.vector_db.add(
            "test2",
            np.array([0.8, 0.2, 0.0]),
            {"name": "test2"}
        )
        self.vector_db.add(
            "test3",
            np.array([0.0, 0.0, 1.0]),
            {"name": "test3"}
        )
        
        # Search for similar vector
        query = np.array([0.9, 0.1, 0.0])
        results = self.vector_db.search(query, top_k=2)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "test1")  # Most similar
        self.assertEqual(results[1][0], "test2")  # Second most similar
    
    def test_remove(self):
        """Test removing vectors."""
        # Add test vector
        self.vector_db.add(
            "test1",
            np.array([1.0, 0.0, 0.0]),
            {"name": "test1"}
        )
        
        # Check it exists
        retrieved_vector, _ = self.vector_db.get("test1")
        self.assertIsNotNone(retrieved_vector)
        
        # Remove it
        self.vector_db.remove("test1")
        
        # Check it's gone
        retrieved_vector, _ = self.vector_db.get("test1")
        self.assertIsNone(retrieved_vector)
    
    def test_save_load(self):
        """Test saving and loading the database."""
        # Add test vectors
        self.vector_db.add(
            "test1",
            np.array([1.0, 0.0, 0.0]),
            {"name": "test1"}
        )
        self.vector_db.add(
            "test2",
            np.array([0.0, 1.0, 0.0]),
            {"name": "test2"}
        )
        
        # Save database
        self.vector_db.save()
        
        # Create new database and load
        new_db = VectorDatabase(self.test_dir.name)
        new_db.load()
        
        # Check loaded data
        vector1, metadata1 = new_db.get("test1")
        self.assertTrue(np.array_equal(vector1, np.array([1.0, 0.0, 0.0])))
        self.assertEqual(metadata1["name"], "test1")
        
        vector2, metadata2 = new_db.get("test2")
        self.assertTrue(np.array_equal(vector2, np.array([0.0, 1.0, 0.0])))
        self.assertEqual(metadata2["name"], "test2")


if __name__ == "__main__":
    unittest.main()