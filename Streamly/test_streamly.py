#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for the Document Chatbot (Qdrant) application.
Tests all components: embeddings, Qdrant connectivity, document processing, and LLM backends.

Run with: python test_streamly.py
"""

import os
import sys
import io
import uuid
import tempfile
import logging
from typing import List
from unittest.mock import Mock, patch, MagicMock

# Test framework
import pytest
import requests

# Add the main script to path for importing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_test_environment():
    """Setup test environment variables"""
    os.environ.setdefault("QDRANT_URL", "https://test-cluster.qdrant.tech:6333")
    os.environ.setdefault("QDRANT_COLLECTION", "test_collection")
    os.environ.setdefault("QDRANT_API_KEY", "test_api_key")
    os.environ.setdefault("HUGGINGFACE_TOKEN", "test_hf_token")

setup_test_environment()

# Import functions from main script
try:
    from streamly import (
        _embedder, _qdrant, _ping_qdrant_rest, _mk_qdrant_client_from_url,
        extract_text_from_upload, chunk_text, upsert_chunks_to_qdrant,
        get_qdrant_context, build_prompt, get_model_response
    )
except ImportError as e:
    print(f"Error importing from streamly.py: {e}")
    print("Make sure streamly.py is in the same directory as this test file")
    sys.exit(1)

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test_header(test_name: str):
    """Print a formatted test header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}Testing: {test_name}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

# ================================
# Test Classes
# ================================

class TestEmbeddings:
    """Test embedding functionality"""
    
    def test_embedder_initialization(self):
        """Test that embedder initializes correctly"""
        print_test_header("Embedding Model Initialization")
        
        try:
            embedder = _embedder()
            print_success("Embedder initialized successfully")
            
            # Test encoding
            test_text = "This is a test sentence for embedding."
            embedding = embedder.encode(test_text)
            
            print_success(f"Generated embedding with shape: {embedding.shape}")
            print_info(f"Embedding dimension: {len(embedding)}")
            
            assert len(embedding) > 0, "Embedding should have non-zero dimension"
            print_success("Embedding generation test passed")
            
        except Exception as e:
            print_error(f"Embedder test failed: {e}")
            return False
        
        return True

class TestQdrantConnectivity:
    """Test Qdrant database connectivity"""
    
    def test_ping_qdrant_rest(self):
        """Test REST ping to Qdrant"""
        print_test_header("Qdrant REST Ping Test")
        
        # Test with valid URL format
        original_url = os.environ.get("QDRANT_URL")
        os.environ["QDRANT_URL"] = "https://test-cluster.qdrant.tech:6333"
        
        try:
            success, message = _ping_qdrant_rest()
            print_info(f"Ping result: {message}")
            
            if success:
                print_success("Qdrant REST ping successful")
            else:
                print_warning(f"Qdrant REST ping failed: {message}")
            
        except Exception as e:
            print_error(f"Ping test error: {e}")
            return False
        finally:
            if original_url:
                os.environ["QDRANT_URL"] = original_url
        
        # Test with missing URL
        print_info("Testing missing URL scenario...")
        del os.environ["QDRANT_URL"]
        success, message = _ping_qdrant_rest()
        assert not success, "Should fail with missing URL"
        print_success("Missing URL test passed")
        
        # Restore original URL
        if original_url:
            os.environ["QDRANT_URL"] = original_url
        
        return True
    
    @patch('streamly.QdrantClient')
    def test_qdrant_client_creation(self, mock_qdrant_client):
        """Test Qdrant client creation with mocking"""
        print_test_header("Qdrant Client Creation")
        
        # Setup mock
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = []
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        
        try:
            os.environ["QDRANT_URL"] = "https://test-cluster.qdrant.tech:6333"
            client, host, port, ipv4s = _mk_qdrant_client_from_url()
            
            print_success(f"Client created for host: {host}, port: {port}")
            print_info(f"Resolved IPv4s: {ipv4s}")
            
            assert host == "test-cluster.qdrant.tech"
            assert port == 6333
            print_success("Client creation test passed")
            
        except Exception as e:
            print_error(f"Client creation test failed: {e}")
            return False
        
        return True

class TestDocumentProcessing:
    """Test document processing functions"""
    
    def test_text_extraction(self):
        """Test text extraction from different file types"""
        print_test_header("Document Text Extraction")
        
        # Test TXT file extraction
        try:
            # Create mock uploaded file
            txt_content = "This is a test document with some sample text."
            txt_file = io.BytesIO(txt_content.encode('utf-8'))
            txt_file.name = "test.txt"
            
            extracted_text = extract_text_from_upload(txt_file)
            assert extracted_text == txt_content, "TXT extraction failed"
            print_success("TXT file extraction test passed")
            
        except Exception as e:
            print_error(f"TXT extraction test failed: {e}")
            return False
        
        # Test PDF extraction (mocked)
        try:
            with patch('streamly.PyPDF2.PdfReader') as mock_pdf_reader:
                # Setup mock PDF reader
                mock_reader = MagicMock()
                mock_page = MagicMock()
                mock_page.extract_text.return_value = "PDF content extracted"
                mock_reader.pages = [mock_page]
                mock_pdf_reader.return_value = mock_reader
                
                pdf_file = io.BytesIO(b"fake pdf content")
                pdf_file.name = "test.pdf"
                
                extracted_text = extract_text_from_upload(pdf_file)
                assert "PDF content extracted" in extracted_text, "PDF extraction failed"
                print_success("PDF file extraction test passed")
                
        except Exception as e:
            print_error(f"PDF extraction test failed: {e}")
            return False
        
        return True
    
    def test_text_chunking(self):
        """Test text chunking functionality"""
        print_test_header("Text Chunking")
        
        try:
            # Test with sample text
            sample_text = "This is a very long document. " * 100  # 3000+ characters
            chunks = chunk_text(sample_text, size=500, overlap=100)
            
            print_success(f"Generated {len(chunks)} chunks from {len(sample_text)} characters")
            
            # Verify chunks
            assert len(chunks) > 1, "Should generate multiple chunks"
            assert all(len(chunk) <= 600 for chunk in chunks), "Chunks too large"  # Allow some margin
            assert all(chunk.strip() for chunk in chunks), "All chunks should be non-empty"
            
            print_info(f"First chunk length: {len(chunks[0])}")
            print_info(f"Last chunk length: {len(chunks[-1])}")
            print_success("Text chunking test passed")
            
        except Exception as e:
            print_error(f"Text chunking test failed: {e}")
            return False
        
        return True
    
    @patch('streamly._qdrant')
    @patch('streamly._embedder')
    def test_upsert_chunks_to_qdrant(self, mock_embedder, mock_qdrant):
        """Test upserting chunks to Qdrant"""
        print_test_header("Upsert Chunks to Qdrant")
        
        try:
            # Setup mocks
            mock_embedder_instance = MagicMock()
            mock_embedder_instance.encode.return_value = [[0.1, 0.2, 0.3]] * 3  # 3 chunks
            mock_embedder.return_value = mock_embedder_instance
            
            mock_client = MagicMock()
            mock_client.upsert.return_value = None
            mock_qdrant.return_value = (mock_client, "test_collection", 384)
            
            # Test upserting
            test_chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
            result = upsert_chunks_to_qdrant(test_chunks, "test_document.txt")
            
            print_success(f"Upserted {result} chunks successfully")
            assert result == 3, "Should return correct number of upserted chunks"
            
            # Verify mocks were called
            mock_embedder_instance.encode.assert_called_once()
            mock_client.upsert.assert_called_once()
            
            print_success("Upsert chunks test passed")
            
        except Exception as e:
            print_error(f"Upsert chunks test failed: {e}")
            return False
        
        return True

class TestRAGFunctionality:
    """Test RAG (Retrieval-Augmented Generation) functionality"""
    
    @patch('streamly._qdrant')
    @patch('streamly._embedder')
    def test_qdrant_context_retrieval(self, mock_embedder, mock_qdrant):
        """Test context retrieval from Qdrant"""
        print_test_header("Qdrant Context Retrieval")
        
        try:
            # Setup mocks
            mock_embedder_instance = MagicMock()
            mock_embedder_instance.encode.return_value = [0.1, 0.2, 0.3]
            mock_embedder.return_value = mock_embedder_instance
            
            mock_client = MagicMock()
            mock_hit1 = MagicMock()
            mock_hit1.score = 0.95
            mock_hit1.payload = {"text": "This is relevant context about the query."}
            
            mock_hit2 = MagicMock()
            mock_hit2.score = 0.87
            mock_hit2.payload = {"text": "This is additional relevant information."}
            
            mock_client.search.return_value = [mock_hit1, mock_hit2]
            mock_qdrant.return_value = (mock_client, "test_collection", 384)
            
            # Test context retrieval
            query = "What is the main topic?"
            context = get_qdrant_context(query, top_k=2)
            
            print_success("Context retrieved successfully")
            print_info(f"Context length: {len(context)} characters")
            
            assert "relevant context" in context, "Context should contain expected content"
            assert "score=0.950" in context, "Context should include scores"
            print_success("Context retrieval test passed")
            
        except Exception as e:
            print_error(f"Context retrieval test failed: {e}")
            return False
        
        return True
    
    def test_prompt_building(self):
        """Test prompt building for RAG"""
        print_test_header("Prompt Building")
        
        try:
            user_question = "What are the benefits of renewable energy?"
            context = "Solar and wind power are clean energy sources. They reduce carbon emissions."
            
            prompt = build_prompt(user_question, context)
            
            print_success("Prompt built successfully")
            print_info(f"Prompt length: {len(prompt)} characters")
            
            assert user_question in prompt, "Prompt should contain user question"
            assert context in prompt, "Prompt should contain context"
            assert "helpful consultant" in prompt.lower(), "Prompt should include role"
            
            print_success("Prompt building test passed")
            
        except Exception as e:
            print_error(f"Prompt building test failed: {e}")
            return False
        
        return True

class TestLLMBackends:
    """Test different LLM backend integrations"""
    
    @patch('streamly.InferenceClient')
    def test_hf_pro_models(self, mock_inference_client):
        """Test Hugging Face Pro models"""
        print_test_header("HF Pro Models Backend")
        
        try:
            # Setup mock for successful chat completion
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "This is a test response from HF Pro model."
            mock_client.chat_completion.return_value = mock_response
            mock_inference_client.return_value = mock_client
            
            # Set environment variable
            os.environ["HUGGINGFACE_TOKEN"] = "test_token"
            
            prompt = "Test prompt for HF Pro models"
            response = get_model_response(prompt, "HF Pro Models")
            
            print_success("HF Pro model response received")
            print_info(f"Response: {response[:100]}...")
            
            assert "test response" in response.lower(), "Should receive expected response"
            print_success("HF Pro models test passed")
            
        except Exception as e:
            print_error(f"HF Pro models test failed: {e}")
            return False
        
        return True
    
    @patch('streamly.InferenceClient')
    def test_hf_third_party_providers(self, mock_inference_client):
        """Test HF Third-Party Providers"""
        print_test_header("HF Third-Party Providers Backend")
        
        try:
            # Setup mock for successful provider response
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response from third-party provider via HF."
            mock_client.chat_completion.return_value = mock_response
            mock_inference_client.return_value = mock_client
            
            os.environ["HUGGINGFACE_TOKEN"] = "test_token"
            
            prompt = "Test prompt for third-party providers"
            response = get_model_response(prompt, "HF Third-Party Providers")
            
            print_success("Third-party provider response received")
            print_info(f"Response: {response[:100]}...")
            
            assert len(response) > 10, "Should receive meaningful response"
            print_success("HF Third-Party Providers test passed")
            
        except Exception as e:
            print_error(f"HF Third-Party Providers test failed: {e}")
            return False
        
        return True
    
    @patch('streamly.OpenAI')
    def test_deepseek_r1(self, mock_openai):
        """Test DeepSeek R1 backend"""
        print_test_header("DeepSeek R1 Backend")
        
        try:
            # Setup mock for OpenAI client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "DeepSeek R1 response to your query."
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            os.environ["OPENROUTER_API_KEY"] = "test_openrouter_key"
            
            prompt = "Test prompt for DeepSeek R1"
            response = get_model_response(prompt, "DeepSeek R1 (cloud)")
            
            print_success("DeepSeek R1 response received")
            print_info(f"Response: {response}")
            
            assert "deepseek" in response.lower(), "Should receive DeepSeek response"
            print_success("DeepSeek R1 test passed")
            
        except Exception as e:
            print_error(f"DeepSeek R1 test failed: {e}")
            return False
        
        return True
    
    def test_local_offline_backend(self):
        """Test Local/Offline backend"""
        print_test_header("Local/Offline Backend")
        
        try:
            # Test with context
            prompt_with_context = """You are a helpful consultant.
Use ONLY the provided context to answer. If the context is insufficient, ask a brief follow-up question.

Context:
[score=0.950] This document discusses renewable energy benefits including reduced emissions.
[score=0.887] Solar power is cost-effective and environmentally friendly.

User question: What are the benefits of renewable energy?
Answer:"""
            
            response = get_model_response(prompt_with_context, "Local/Offline")
            
            print_success("Local/Offline response received")
            print_info(f"Response: {response}")
            
            assert "renewable energy" in response.lower(), "Should mention renewable energy"
            assert len(response) > 20, "Should provide meaningful response"
            print_success("Local/Offline test passed")
            
        except Exception as e:
            print_error(f"Local/Offline test failed: {e}")
            return False
        
        return True

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_missing_environment_variables(self):
        """Test behavior with missing environment variables"""
        print_test_header("Missing Environment Variables")
        
        try:
            # Backup original values
            original_hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            original_openrouter = os.environ.get("OPENROUTER_API_KEY")
            
            # Remove tokens
            if "HUGGINGFACE_TOKEN" in os.environ:
                del os.environ["HUGGINGFACE_TOKEN"]
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            
            # Test HF Pro without token
            response = get_model_response("test", "HF Pro Models")
            assert "HUGGINGFACE_TOKEN" in response, "Should ask for HF token"
            print_success("Missing HF token handling works")
            
            # Test DeepSeek without token
            response = get_model_response("test", "DeepSeek R1 (cloud)")
            assert "OPENROUTER_API_KEY" in response, "Should ask for OpenRouter key"
            print_success("Missing OpenRouter key handling works")
            
            # Restore original values
            if original_hf_token:
                os.environ["HUGGINGFACE_TOKEN"] = original_hf_token
            if original_openrouter:
                os.environ["OPENROUTER_API_KEY"] = original_openrouter
            
            print_success("Missing environment variables test passed")
            
        except Exception as e:
            print_error(f"Missing environment variables test failed: {e}")
            return False
        
        return True
    
    def test_invalid_model_choice(self):
        """Test invalid model choice handling"""
        print_test_header("Invalid Model Choice")
        
        try:
            response = get_model_response("test prompt", "NonExistentModel")
            assert "not implemented" in response, "Should handle invalid model choice"
            print_success("Invalid model choice handling works")
            
        except Exception as e:
            print_error(f"Invalid model choice test failed: {e}")
            return False
        
        return True

# ================================
# Main Test Runner
# ================================

def run_all_tests():
    """Run all tests and provide summary"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 80)
    print("🚀 DOCUMENT CHATBOT COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"{Colors.END}")
    
    test_classes = [
        TestEmbeddings(),
        TestQdrantConnectivity(),
        TestDocumentProcessing(),
        TestRAGFunctionality(),
        TestLLMBackends(),
        TestErrorHandling(),
    ]
    
    results = {}
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            test_method = getattr(test_class, method_name)
            
            try:
                success = test_method()
                if success:
                    passed_tests += 1
                    results[f"{class_name}.{method_name}"] = "PASSED"
                else:
                    results[f"{class_name}.{method_name}"] = "FAILED"
            except Exception as e:
                results[f"{class_name}.{method_name}"] = f"ERROR: {e}"
    
    # Print summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"{Colors.END}")
    
    for test_name, result in results.items():
        if result == "PASSED":
            print_success(f"{test_name}: {result}")
        else:
            print_error(f"{test_name}: {result}")
    
    print(f"\n{Colors.BOLD}")
    if passed_tests == total_tests:
        print_success(f"ALL TESTS PASSED! ({passed_tests}/{total_tests})")
    else:
        print_error(f"SOME TESTS FAILED: {passed_tests}/{total_tests} passed")
    print(f"{Colors.END}")
    
    # Recommendations
    print(f"\n{Colors.BOLD}{Colors.YELLOW}")
    print("🔧 RECOMMENDATIONS:")
    print(f"{Colors.END}")
    
    if passed_tests < total_tests:
        print_warning("Some tests failed. Check the errors above and:")
        print_info("1. Ensure all required packages are installed")
        print_info("2. Check environment variables are set correctly")
        print_info("3. Verify network connectivity to external services")
    
    if "HUGGINGFACE_TOKEN" not in os.environ:
        print_warning("Set HUGGINGFACE_TOKEN for full HF testing")
    
    if "OPENROUTER_API_KEY" not in os.environ:
        print_info("Set OPENROUTER_API_KEY to test DeepSeek integration")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # Setup logging to reduce noise during testing
    logging.getLogger().setLevel(logging.ERROR)
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)