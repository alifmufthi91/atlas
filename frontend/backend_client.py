import requests
import json
from typing import List, Dict, Any, Optional
import base64

class BackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def set_auth_token(self, token: str):
        """Set authentication token"""
        self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def get_google_auth_url(self) -> str:
        """Get Google OAuth URL with state parameter"""
        response = self.session.get(f"{self.base_url}/auth/google-url?state=gmail")
        response.raise_for_status()
        return response.json()["auth_url"]
    
    def authenticate_with_google(self, code: str) -> Dict[str, str]:
        """Authenticate with Google OAuth code"""
        response = self.session.post(
            f"{self.base_url}/auth/google-callback",
            json={"code": code}
        )
        response.raise_for_status()
        return response.json()
    
    def authenticate_with_twitter(self, code: str) -> Dict[str, str]:
        """Authenticate with Twitter OAuth code"""
        response = self.session.post(
            f"{self.base_url}/auth/twitter-callback",
            json={"code": code}
        )
        response.raise_for_status()
        return response.json()
    
    def chat(self, messages: List[Dict[str, str]], session_id: str, thread_id: str = None, auth_tokens: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Send chat request with thread ID"""
        payload = {
            "messages": messages, 
            "session_id": session_id,
            "thread_id": thread_id,
            "auth_tokens": auth_tokens or {}
        }
        response = self.session.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    
    def upload_pdf_anonymous(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload PDF file without authentication"""
        file_b64 = base64.b64encode(file_content).decode()
        response = self.session.post(
            f"{self.base_url}/upload-pdf-anonymous",
            json={"file_content": file_b64, "filename": filename}
        )
        response.raise_for_status()
        return response.json()
    
    def logout_provider(self, provider: str, token: str):
        """Logout from specific provider"""
        response = self.session.delete(
            f"{self.base_url}/logout/{provider}",
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        return response.json()
    
    def get_session_threads(self, session_id: str) -> Dict[str, Any]:
        """Get all threads for a session"""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/threads")
        response.raise_for_status()
        return response.json()
    
    def create_new_thread(self, session_id: str) -> Dict[str, Any]:
        """Create a new thread"""
        response = self.session.post(f"{self.base_url}/sessions/{session_id}/threads/new")
        response.raise_for_status()
        return response.json()
    
    def delete_thread(self, session_id: str, thread_id: str) -> Dict[str, Any]:
        """Delete a thread"""
        response = self.session.delete(f"{self.base_url}/sessions/{session_id}/threads/{thread_id}")
        response.raise_for_status()
        return response.json()
    
    def validate_tokens(self, tokens: Dict[str, str]) -> Dict[str, Any]:
        """Validate tokens with backend"""
        response = self.session.post(f"{self.base_url}/validate-tokens", json=tokens)
        response.raise_for_status()
        return response.json()