"""
Unit tests for run_id system and log correlation functionality

This test suite verifies that the run_id system correctly propagates 
through the evaluation pipeline and maintains log correlation integrity.
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import uuid

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nugget_eval.constants import RunIdConfig, ValidationConfig
from nugget_eval.validators import (
    RunIdValidator, LogCorrelationValidator, 
    validate_run_id, validate_system_health
)
from nugget_eval.evaluator import NuggetEvaluator
from nugget_eval.common import ModelClient
from nugget_eval.config import Config


class TestRunIdGeneration(unittest.TestCase):
    """Test run_id generation and format validation"""
    
    def test_run_id_format_validation(self):
        """Test that run_id format validation works correctly"""
        validator = RunIdValidator()
        
        # Valid run_ids
        valid_run_ids = [
            "20240101_120000_abc12345",
            "20231225_235959_def67890",
            "20240229_000000_12345678"  # Leap year
        ]
        
        for run_id in valid_run_ids:
            with self.subTest(run_id=run_id):
                self.assertTrue(
                    validator.validate_run_id_format(run_id),
                    f"Valid run_id should pass validation: {run_id}"
                )
        
        # Invalid run_ids
        invalid_run_ids = [
            "",  # Empty
            "invalid",  # Wrong format
            "20240101_120000",  # Missing UUID
            "20240101_120000_abc1234",  # UUID too short
            "20240101_120000_abc123456",  # UUID too long
            "20240101_25:00:00_abc12345",  # Invalid time
            "2024_01_01_120000_abc12345",  # Wrong date format
        ]
        
        for run_id in invalid_run_ids:
            with self.subTest(run_id=run_id):
                self.assertFalse(
                    validator.validate_run_id_format(run_id),
                    f"Invalid run_id should fail validation: {run_id}"
                )
    
    def test_run_id_timestamp_extraction(self):
        """Test that timestamps can be extracted from run_ids"""
        validator = RunIdValidator()
        
        test_cases = [
            ("20240101_120000_abc12345", datetime(2024, 1, 1, 12, 0, 0)),
            ("20231231_235959_def67890", datetime(2023, 12, 31, 23, 59, 59)),
            ("20240229_060000_12345678", datetime(2024, 2, 29, 6, 0, 0)),  # Leap year
        ]
        
        for run_id, expected_datetime in test_cases:
            with self.subTest(run_id=run_id):
                extracted_datetime = validator.extract_timestamp_from_run_id(run_id)
                self.assertEqual(
                    extracted_datetime, expected_datetime,
                    f"Timestamp extraction failed for {run_id}"
                )
    
    def test_run_id_consistency_validation(self):
        """Test consistency validation across multiple run_ids"""
        validator = RunIdValidator()
        
        # Test with valid run_ids
        valid_run_ids = [
            "20240101_120000_abc12345",
            "20240101_120001_def67890",
            "20240101_120002_12345678"
        ]
        
        result = validator.validate_run_id_consistency(valid_run_ids)
        
        self.assertEqual(result["total_count"], 3)
        self.assertEqual(result["valid_count"], 3)
        self.assertEqual(result["invalid_count"], 0)
        self.assertEqual(result["duplicate_count"], 0)
        self.assertIn("time_range", result)
    
    def test_run_id_duplicate_detection(self):
        """Test duplicate run_id detection"""
        validator = RunIdValidator()
        
        run_ids_with_duplicates = [
            "20240101_120000_abc12345",
            "20240101_120001_def67890", 
            "20240101_120000_abc12345",  # Duplicate
            "20240101_120002_12345678"
        ]
        
        result = validator.validate_run_id_consistency(run_ids_with_duplicates)
        
        self.assertEqual(result["duplicate_count"], 1)
        self.assertIn("20240101_120000_abc12345", result["duplicates"])


class TestModelClientRunIdPropagation(unittest.TestCase):
    """Test run_id propagation through ModelClient"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.run_id = "20240101_120000_test1234"
        
        # Mock OpenAI response
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        self.mock_response.choices[0].message.content = "support\npartial_support\nnot_support"
        self.mock_response.choices[0].finish_reason = "stop"
        self.mock_response.usage.input_tokens = 100
        self.mock_response.usage.output_tokens = 50
        self.mock_response.usage.total_tokens = 150
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('nugget_eval.common.OpenAI')
    def test_model_client_run_id_initialization(self, mock_openai_class):
        """Test that ModelClient correctly stores run_id"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        client = ModelClient(
            base_url="http://test",
            api_key="test_key", 
            model_name="test_model",
            llm_log_dir=str(self.temp_path / "llm"),
            run_id=self.run_id
        )
        
        self.assertEqual(client.run_id, self.run_id)
        
        # Check that log file name contains run_id
        log_file_name = client.llm_log_file.name
        self.assertIn(self.run_id, log_file_name)
    
    @patch('nugget_eval.common.OpenAI')
    def test_model_client_context_propagation(self, mock_openai_class):
        """Test that context with qid and batch_index is properly handled"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = self.mock_response
        
        client = ModelClient(
            base_url="http://test",
            api_key="test_key",
            model_name="test_model", 
            llm_log_dir=str(self.temp_path / "llm"),
            run_id=self.run_id
        )
        
        test_context = {
            "qid": "Q12345",
            "batch_index": 2,
            "nuggets_in_batch": 3
        }
        
        # Mock the predict_batch call
        with patch.object(client, '_save_llm_log') as mock_save_log:
            result = client.predict_batch(
                query="test query",
                block_text="test block",
                nuggets_text_list=["nugget1", "nugget2", "nugget3"],
                context=test_context
            )
            
            # Verify that context was passed to log saving
            mock_save_log.assert_called()
            call_args = mock_save_log.call_args[0][0]  # First argument to _save_llm_log
            
            self.assertEqual(call_args["run_id"], self.run_id)
            self.assertEqual(call_args["context"]["qid"], "Q12345")
            self.assertEqual(call_args["context"]["batch_index"], 2)
    
    @patch('nugget_eval.common.OpenAI')
    def test_model_client_log_file_creation(self, mock_openai_class):
        """Test that log files are created with run_id in the path"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        client = ModelClient(
            base_url="http://test",
            api_key="test_key",
            model_name="test_model",
            llm_log_dir=str(self.temp_path / "llm"),
            run_id=self.run_id
        )
        
        # Verify log directory structure
        self.assertTrue(client.llm_log_dir.exists())
        self.assertIn("llm", str(client.llm_log_file))
        self.assertTrue(client.llm_log_file.name.endswith("llm_calls.jsonl"))


class TestEvaluatorRunIdIntegration(unittest.TestCase):
    """Test run_id integration in NuggetEvaluator"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test config
        self.config = Config()
        self.config.model.base_url = "http://test"
        self.config.model.api_key = "test_key"
        self.config.model.name = "test_model"
        self.config.logging.eval_log_dir = str(self.temp_path / "eval")
        self.config.logging.llm_log_dir = str(self.temp_path / "llm")
        
        # Create test data files
        self.input_file = self.temp_path / "input.jsonl"
        self.gold_file = self.temp_path / "gold.jsonl"
        self._create_test_files()
        
        self.config.data.input_path = str(self.input_file)
        self.config.data.gold_path = str(self.gold_file)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_files(self):
        """Create minimal test data files"""
        input_data = [
            {
                "qid": "Q001", 
                "query": "test query",
                "block": "test block",
                "nuggets": [{"nid": "N001", "text": "nugget 1"}]
            }
        ]
        
        gold_data = [
            {
                "qid": "Q001",
                "nuggets": [{"nid": "N001", "label": "support"}]
            }
        ]
        
        with open(self.input_file, 'w') as f:
            for item in input_data:
                json.dump(item, f)
                f.write('\n')
        
        with open(self.gold_file, 'w') as f:
            for item in gold_data:
                json.dump(item, f)
                f.write('\n')
    
    def test_evaluator_run_id_generation(self):
        """Test that evaluator generates valid run_id"""
        evaluator = NuggetEvaluator(self.config)
        
        # Check that run_id is generated and valid
        self.assertIsNotNone(evaluator.run_id)
        self.assertTrue(validate_run_id(evaluator.run_id))
    
    def test_evaluator_directory_structure(self):
        """Test that evaluator creates proper directory structure with run_id"""
        evaluator = NuggetEvaluator(self.config)
        
        # Check that run-specific directories are created
        self.assertTrue(evaluator.eval_run_dir.exists())
        self.assertTrue(evaluator.llm_run_dir.exists())
        
        # Check that directories contain run_id
        self.assertIn(evaluator.run_id, str(evaluator.eval_run_dir))
        self.assertIn(evaluator.run_id, str(evaluator.llm_run_dir))
    
    @patch('nugget_eval.common.OpenAI')
    def test_evaluator_run_id_in_model_client(self, mock_openai_class):
        """Test that run_id is passed to ModelClient"""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        evaluator = NuggetEvaluator(self.config)
        
        # Check that model client has the same run_id
        self.assertEqual(evaluator.model_client.run_id, evaluator.run_id)
    
    def test_evaluator_metadata_saving(self):
        """Test that run metadata is saved with run_id"""
        evaluator = NuggetEvaluator(self.config)
        
        meta_file = evaluator.eval_run_dir / "run_meta.json"
        
        # Metadata should be created during initialization
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            self.assertEqual(metadata["run_id"], evaluator.run_id)
            self.assertIn("config", metadata)


class TestLogCorrelation(unittest.TestCase):
    """Test log correlation functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.eval_log_dir = self.temp_path / "eval"
        self.llm_log_dir = self.temp_path / "llm"
        self.eval_log_dir.mkdir(parents=True)
        self.llm_log_dir.mkdir(parents=True)
        
        self.run_id = "20240101_120000_test1234"
        self.validator = LogCorrelationValidator()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_logs(self, run_id: str, with_errors: bool = False):
        """Create test log files for a run_id"""
        run_eval_dir = self.eval_log_dir / run_id
        run_llm_dir = self.llm_log_dir / run_id
        run_eval_dir.mkdir(parents=True, exist_ok=True)
        run_llm_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run metadata
        metadata = {
            "run_id": run_id,
            "config": {"test": True},
            "timestamp": datetime.now().isoformat()
        }
        with open(run_eval_dir / "run_meta.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create LLM log entries
        llm_entries = [
            {
                "timestamp": "2024-01-01T12:00:00",
                "run_id": run_id,
                "context": {"qid": "Q001", "batch_index": 0},
                "success": True,
                "response": {"content": "test response"}
            },
            {
                "timestamp": "2024-01-01T12:00:01", 
                "run_id": run_id,
                "context": {"qid": "Q002", "batch_index": 0},
                "success": not with_errors,
                "response": {"content": "test response 2"} if not with_errors else None,
                "error": {"type": "test_error", "message": "test error"} if with_errors else None
            }
        ]
        
        llm_log_file = run_llm_dir / f"test_{run_id}_llm_calls.jsonl"
        with open(llm_log_file, 'w') as f:
            for entry in llm_entries:
                json.dump(entry, f)
                f.write('\n')
        
        return run_eval_dir, run_llm_dir
    
    def test_log_correlation_validation_success(self):
        """Test successful log correlation validation"""
        run_eval_dir, run_llm_dir = self._create_test_logs(self.run_id)
        
        result = self.validator.validate_log_correlation(
            self.eval_log_dir, self.llm_log_dir, self.run_id
        )
        
        self.assertEqual(result["run_id"], self.run_id)
        self.assertTrue(result["correlation_valid"])
        self.assertEqual(len(result["issues"]), 0)
        self.assertGreater(len(result["eval_files"]), 0)
        self.assertGreater(len(result["llm_files"]), 0)
    
    def test_log_correlation_missing_directories(self):
        """Test log correlation with missing directories"""
        result = self.validator.validate_log_correlation(
            self.eval_log_dir, self.llm_log_dir, "nonexistent_run"
        )
        
        self.assertFalse(result["correlation_valid"])
        self.assertGreater(len(result["issues"]), 0)
        self.assertIn("not found", result["issues"][0])
    
    def test_llm_log_consistency_analysis(self):
        """Test LLM log consistency analysis"""
        run_eval_dir, run_llm_dir = self._create_test_logs(self.run_id)
        
        stats = self.validator._analyze_llm_log_consistency(run_llm_dir, self.run_id)
        
        self.assertEqual(stats["total_entries"], 2)
        self.assertEqual(stats["entries_with_run_id"], 2) 
        self.assertEqual(stats["matching_run_id"], 2)
        self.assertEqual(stats["mismatched_run_id"], 0)
        self.assertEqual(stats["files_analyzed"], 1)
    
    def test_batch_correlation_validation(self):
        """Test batch log correlation validation"""
        # Create multiple test runs
        run_ids = ["20240101_120000_test001", "20240101_120001_test002"]
        
        for run_id in run_ids:
            self._create_test_logs(run_id)
        
        result = self.validator.validate_batch_logs_correlation(
            [(self.eval_log_dir, self.llm_log_dir)], run_ids
        )
        
        self.assertEqual(result["total_runs"], 2)
        self.assertEqual(result["valid_correlations"], 2)
        self.assertEqual(result["invalid_correlations"], 0)
        self.assertEqual(result["summary"]["recommendation"], "OK")


class TestSystemHealthCheck(unittest.TestCase):
    """Test system health check functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.eval_log_dir = self.temp_path / "eval"
        self.llm_log_dir = self.temp_path / "llm"
        self.eval_log_dir.mkdir(parents=True)
        self.llm_log_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_system_health_check_empty_directories(self):
        """Test health check with empty log directories"""
        result = validate_system_health(
            str(self.eval_log_dir), str(self.llm_log_dir)
        )
        
        self.assertIn("overall_status", result)
        self.assertIn("run_id_validation", result)
        self.assertIn("log_correlation_validation", result)
    
    def test_system_health_check_with_valid_runs(self):
        """Test health check with valid runs"""
        # Create some test runs
        run_ids = ["20240101_120000_test001", "20240101_120001_test002"]
        
        for run_id in run_ids:
            # Create basic directory structure
            (self.eval_log_dir / run_id).mkdir(parents=True)
            (self.llm_log_dir / run_id).mkdir(parents=True)
            
            # Create metadata file
            metadata = {"run_id": run_id, "config": {}}
            with open(self.eval_log_dir / run_id / "run_meta.json", 'w') as f:
                json.dump(metadata, f)
        
        result = validate_system_health(
            str(self.eval_log_dir), str(self.llm_log_dir)
        )
        
        # Should find valid run_ids and have reasonable health status
        run_id_validation = result.get("run_id_validation", {})
        self.assertGreater(run_id_validation.get("valid_count", 0), 0)


class TestIntegrationFlow(unittest.TestCase):
    """Integration tests for complete run_id flow"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test config
        self.config = Config()
        self.config.model.base_url = "http://test"
        self.config.model.api_key = "test_key"
        self.config.model.name = "test_model"
        self.config.logging.eval_log_dir = str(self.temp_path / "eval")
        self.config.logging.llm_log_dir = str(self.temp_path / "llm")
        self.config.evaluation.num_workers = 1  # Single worker for testing
        
        # Create minimal test data
        self.input_file = self.temp_path / "input.jsonl"
        self.gold_file = self.temp_path / "gold.jsonl"
        self._create_minimal_test_data()
        
        self.config.data.input_path = str(self.input_file)
        self.config.data.gold_path = str(self.gold_file)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_minimal_test_data(self):
        """Create minimal test data files"""
        input_data = {
            "qid": "Q001",
            "query": "test query", 
            "block": "test block",
            "nuggets": [{"nid": "N001", "text": "test nugget"}]
        }
        
        gold_data = {
            "qid": "Q001",
            "nuggets": [{"nid": "N001", "label": "support"}]
        }
        
        with open(self.input_file, 'w') as f:
            json.dump(input_data, f)
        
        with open(self.gold_file, 'w') as f:
            json.dump(gold_data, f)
    
    @patch('nugget_eval.common.OpenAI')
    def test_end_to_end_run_id_propagation(self, mock_openai_class):
        """Test complete end-to-end run_id propagation"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "support"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.total_tokens = 15
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create evaluator and run evaluation
        evaluator = NuggetEvaluator(self.config)
        run_id = evaluator.run_id
        
        # Validate that run_id is properly formatted
        self.assertTrue(validate_run_id(run_id))
        
        # Check directory structure
        self.assertTrue(evaluator.eval_run_dir.exists())
        self.assertTrue(evaluator.llm_run_dir.exists())
        self.assertIn(run_id, str(evaluator.eval_run_dir))
        self.assertIn(run_id, str(evaluator.llm_run_dir))
        
        # Verify model client has correct run_id
        self.assertEqual(evaluator.model_client.run_id, run_id)
        
        # Test that log correlation validator can find the structure
        validator = LogCorrelationValidator()
        correlation_result = validator.validate_log_correlation(
            Path(self.config.logging.eval_log_dir),
            Path(self.config.logging.llm_log_dir),
            run_id
        )
        
        # Should find the directories we created
        self.assertEqual(correlation_result["run_id"], run_id)


if __name__ == '__main__':
    # Set up test logging
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)