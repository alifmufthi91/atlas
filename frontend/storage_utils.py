import streamlit as st
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pickle
import os

class PersistentStorage:
    """Simple file-based storage that persists across sessions"""
    
    def __init__(self):
        self.storage_dir = ".streamlit_storage"
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def _get_file_path(self, key: str) -> str:
        return os.path.join(self.storage_dir, f"{key}.json")
    
    def set_item(self, key: str, value: Any, ttl_hours: int = 24):
        """Set item with TTL"""
        expiry = datetime.now() + timedelta(hours=ttl_hours)
        data = {
            "value": value,
            "expiry": expiry.isoformat()
        }
        
        try:
            with open(self._get_file_path(key), 'w') as f:
                json.dump(data, f)
        except Exception as e:
            st.error(f"Failed to save {key}: {e}")
    
    def get_item(self, key: str) -> Optional[Any]:
        """Get item with TTL check"""
        file_path = self._get_file_path(key)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            expiry = datetime.fromisoformat(data["expiry"])
            if datetime.now() > expiry:
                # Expired, remove it
                self.remove_item(key)
                return None
            
            return data["value"]
        except Exception as e:
            # Invalid data, remove it
            self.remove_item(key)
            return None
    
    def remove_item(self, key: str):
        """Remove item"""
        file_path = self._get_file_path(key)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                st.error(f"Failed to remove {key}: {e}")

# Global storage instance
storage = PersistentStorage()

class LocalStorageManager:
    """Manager class for persistent storage operations"""
    
    @staticmethod
    def set_gmail_token(token: str, user_id: str):
        """Store Gmail token with 1 day TTL"""
        token_data = {
            "access_token": token,
            "user_id": user_id,
            "provider": "gmail",
            "created_at": datetime.now().isoformat()
        }
        
        # Store in file
        storage.set_item("gmail_token", token_data, ttl_hours=24)
        
        # Also set in session state for immediate use
        st.session_state["gmail_access_token"] = token
        st.session_state["gmail_user_id"] = user_id
        
        st.success("✅ Gmail token saved successfully")
    
    @staticmethod
    def get_gmail_token() -> Optional[Dict[str, str]]:
        """Get Gmail token if not expired"""
        # First check session state
        if "gmail_access_token" in st.session_state:
            return {
                "access_token": st.session_state["gmail_access_token"],
                "user_id": st.session_state.get("gmail_user_id", ""),
                "provider": "gmail"
            }
        
        # Then check persistent storage
        token_data = storage.get_item("gmail_token")
        if token_data:
            st.session_state["gmail_access_token"] = token_data["access_token"]
            st.session_state["gmail_user_id"] = token_data["user_id"]
            return token_data
        
        return None
    
    @staticmethod
    def remove_gmail_token():
        """Remove Gmail token"""
        storage.remove_item("gmail_token")
        
        # Clear session state
        if "gmail_access_token" in st.session_state:
            del st.session_state["gmail_access_token"]
        if "gmail_user_id" in st.session_state:
            del st.session_state["gmail_user_id"]
    
    @staticmethod
    def add_recent_thread(thread_id: str, session_id: str, title: str = None):
        """Add thread to recent threads (keep last 3)"""
        recent_threads = storage.get_item("recent_threads") or []
        
        # Remove if already exists
        recent_threads = [t for t in recent_threads if t["thread_id"] != thread_id]
        
        # Add to beginning
        thread_data = {
            "thread_id": thread_id,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "title": title or f"Thread {thread_id[-8:]}"
        }
        recent_threads.insert(0, thread_data)
        
        # Keep only last 3
        recent_threads = recent_threads[:3]
        
        storage.set_item("recent_threads", recent_threads, ttl_hours=24 * 7)  # 1 week TTL
    
    @staticmethod
    def get_recent_threads() -> List[Dict[str, str]]:
        """Get recent threads"""
        return storage.get_item("recent_threads") or []
    
    @staticmethod
    def remove_thread_from_recent(thread_id: str):
        """Remove thread from recent threads"""
        recent_threads = storage.get_item("recent_threads") or []
        recent_threads = [t for t in recent_threads if t["thread_id"] != thread_id]
        storage.set_item("recent_threads", recent_threads, ttl_hours=24 * 7)
    
    @staticmethod
    def update_thread_last_used(thread_id: str):
        """Update thread's last used timestamp"""
        recent_threads = storage.get_item("recent_threads") or []
        for thread in recent_threads:
            if thread["thread_id"] == thread_id:
                thread["last_used"] = datetime.now().isoformat()
                break
        storage.set_item("recent_threads", recent_threads, ttl_hours=24 * 7)
    
    @staticmethod
    def clear_all():
        """Clear all stored data"""
        storage.remove_item("gmail_token")
        storage.remove_item("recent_threads")
        
        # Clear session state
        for key in ["gmail_access_token", "gmail_user_id"]:
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def get_storage_status() -> Dict[str, Any]:
        """Get storage status for debugging"""
        gmail_token = storage.get_item("gmail_token")
        recent_threads = storage.get_item("recent_threads") or []
        
        return {
            "gmail_token_exists": gmail_token is not None,
            "gmail_token_data": gmail_token,
            "recent_threads_count": len(recent_threads),
            "recent_threads": recent_threads,
            "session_state_gmail": "gmail_access_token" in st.session_state,
            "storage_dir": storage.storage_dir
        }