#!/usr/bin/env python3
"""
Unified Transformerlab API Client

This module provides a comprehensive interface to the Transformerlab API,
supporting authentication and all available endpoints from the API documentation.
This supports the Evolutionary Layering in Complex Systems (ELCS) framework
by providing programmatic access to Transformerlab's capabilities.

Author: Claude & angrysky56
Date: April 30, 2025
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("transformerlab_api")

class TransformerlabAPI:
    """
    Unified client for interacting with the Transformerlab API.
    
    This class provides methods for all API endpoints documented at:
    http://localhost:8338/docs
    
    It handles authentication, request formatting, and response parsing.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8338", 
        token: Optional[str] = None
    ):
        """
        Initialize the Transformerlab API client.
        
        Args:
            base_url: The base URL of the Transformerlab API
            token: Optional bearer token for authentication. If not provided,
                   will attempt to load from environment variable TRANSFORMERLAB_API_TOKEN
        """
        self.base_url = base_url.rstrip('/')
        self.token = token or os.environ.get("TRANSFORMERLAB_API_TOKEN")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
        else:
            logger.warning("No API token provided. Some endpoints may not be accessible.")
            
        # Test connection
        try:
            self.test_connection()
            logger.info(f"Successfully connected to Transformerlab API at {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not connect to Transformerlab API: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test connection to the API by making a simple request."""
        response = requests.get(f"{self.base_url}/ping")
        if response.status_code != 200:
            raise ConnectionError(f"Failed to connect to API. Status: {response.status_code}")
        return True
    
    def set_token(self, token: str) -> None:
        """Set or update the authentication token."""
        self.token = token
        self.headers["Authorization"] = f"Bearer {self.token}"
        logger.info("Authentication token updated")
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path (without base URL)
            params: Optional query parameters
            data: Optional JSON body data
            files: Optional files to upload
            
        Returns:
            Parsed JSON response
        """
        url = urljoin(f"{self.base_url}/", endpoint.lstrip('/'))
        
        # Prepare headers - if sending files, don't include Content-Type
        headers = self.headers.copy()
        if files:
            headers.pop("Content-Type", None)
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                if files:
                    # For multipart/form-data with files
                    response = requests.post(url, headers=headers, params=params, 
                                            data=data, files=files)
                else:
                    # For JSON body
                    response = requests.post(url, headers=headers, params=params, 
                                           json=data)
            elif method == "PUT":
                response = requests.put(url, headers=headers, params=params, 
                                       json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params, 
                                         json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            # Check for HTTP errors
            response.raise_for_status()
            
            # Return JSON response if available
            if response.headers.get("Content-Type", "").startswith("application/json"):
                return response.json()
            else:
                return response.text
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response body: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error making request: {str(e)}")
            raise
    
    # ---- DATA/DATASETS ENDPOINTS ----
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        return self._make_request("GET", "/data/list")
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific dataset."""
        return self._make_request("GET", f"/data/info?id={dataset_id}")
    
    def preview_dataset(self, dataset_id: str, num_samples: int = 5) -> Dict[str, Any]:
        """Preview the contents of a dataset."""
        return self._make_request("GET", f"/data/preview?id={dataset_id}&samples={num_samples}")
    
    def preview_with_template(self, dataset_id: str, template_id: str) -> Dict[str, Any]:
        """Preview a dataset with a prompt template applied."""
        return self._make_request("GET", f"/data/preview_with_template?id={dataset_id}&template_id={template_id}")
    
    def download_dataset(self, 
                         dataset_name: str, 
                         subset: Optional[str] = None,
                         split: str = "train") -> Dict[str, Any]:
        """Download a dataset from Hugging Face."""
        params = {
            "name": dataset_name,
            "split": split
        }
        if subset:
            params["subset"] = subset
        
        return self._make_request("GET", "/data/download", params=params)
    
    def list_generated_datasets(self) -> List[Dict[str, Any]]:
        """List available generated datasets."""
        return self._make_request("GET", "/data/generated_datasets_list")
    
    def upload_dataset(self, file_path: str, name: str) -> Dict[str, Any]:
        """Upload a local dataset file."""
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            return self._make_request("POST", "/data/upload", 
                                     data={"name": name}, 
                                     files=files)
    
    # ---- MODEL ENDPOINTS ----
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        return self._make_request("GET", "/model/list")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        return self._make_request("GET", f"/model/info?id={model_id}")
    
    def download_model(self, model_id: str) -> Dict[str, Any]:
        """Download a model from Hugging Face."""
        return self._make_request("GET", f"/model/download?id={model_id}")
    
    # ---- TRAINING