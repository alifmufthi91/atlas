import streamlit as st
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

def set_local_storage_item(key: str, value: Any, ttl_hours: int = 24):
    """Set item in localStorage with TTL"""
    expiry = datetime.now() + timedelta(hours=ttl_hours)
    data = {
        "value": value,
        "expiry": expiry.isoformat()
    }
    
    # Use Streamlit's session state to persist and JavaScript to sync with localStorage
    st.session_state[f"ls_{key}"] = data
    
    # JavaScript to set localStorage
    js_code = f"""
    <script>
        localStorage.setItem('{key}', '{json.dumps(data)}');
    </script>
    """
    st.components.v1.html(js_code, height=0)

def get_local_storage_item(key: str) -> Optional[Any]:
    """Get item from localStorage with TTL check"""
    # First try to get from session state
    data = st.session_state.get(f"ls_{key}")
    
    if data:
        try:
            expiry = datetime.fromisoformat(data["expiry"])
            if datetime.now() > expiry:
                # Expired, remove it
                remove_local_storage_item(key)
                return None
            return data["value"]
        except (KeyError, ValueError):
            # Invalid data format, remove it
            remove_local_storage_item(key)
            return None
    
    return None

def remove_local_storage_item(key: str):
    """Remove item from localStorage"""
    if f"ls_{key}" in st.session_state:
        del st.session_state[f"ls_{key}"]
    
    # JavaScript to remove from localStorage
    js_code = f"""
    <script>
        localStorage.removeItem('{key}');
    </script>
    """
    st.components.v1.html(js_code, height=0)

def initialize_local_storage():
    """Initialize localStorage sync on page load"""
    # JavaScript to sync localStorage with Streamlit session state
    js_code = """
    <script>
        // Function to sync localStorage to Streamlit
        function syncLocalStorageToStreamlit() {
            const keys = ['gmail_token', 'recent_threads'];
            
            keys.forEach(key => {
                const value = localStorage.getItem(key);
                if (value) {
                    try {
                        const data = JSON.parse(value);
                        // Check if expired
                        if (data.expiry && new Date() > new Date(data.expiry)) {
                            localStorage.removeItem(key);
                        } else {
                            // Send to Streamlit via custom event
                            window.parent.postMessage({
                                type: 'localStorage_sync',
                                key: key,
                                data: data
                            }, '*');
                        }
                    } catch (e) {
                        localStorage.removeItem(key);
                    }
                }
            });
        }
        
        // Sync on load
        syncLocalStorageToStreamlit();
        
        // Listen for Streamlit updates
        window.addEventListener('message', function(event) {
            if (event.data.type === 'streamlit_update') {
                const { key, data } = event.data;
                if (data === null) {
                    localStorage.removeItem(key);
                } else {
                    localStorage.setItem(key, JSON.stringify(data));
                }
            }
        });
    </script>
    """
    st.components.v1.html(js_code, height=0)

class LocalStorageManager:
    """Manager class for localStorage operations"""
    
    @staticmethod
    def set_gmail_token(token: str, user_id: str):
        """Store Gmail token with 1 day TTL"""
        token_data = {
            "access_token": token,
            "user_id": user_id,
            "provider": "gmail"
        }
        set_local_storage_item("gmail_token", token_data, ttl_hours=24)
        
        # Also set in session state for immediate use
        st.session_state["gmail_access_token"] = token
        st.session_state["gmail_user_id"] = user_id
    
    @staticmethod
    def get_gmail_token() -> Optional[Dict[str, str]]:
        """Get Gmail token if not expired"""
        token_data = get_local_storage_item("gmail_token")
        if token_data:
            st.session_state["gmail_access_token"] = token_data["access_token"]
            st.session_state["gmail_user_id"] = token_data["user_id"]
            return token_data
        return None
    
    @staticmethod
    def remove_gmail_token():
        """Remove Gmail token"""
        remove_local_storage_item("gmail_token")
        if "gmail_access_token" in st.session_state:
            del st.session_state["gmail_access_token"]
        if "gmail_user_id" in st.session_state:
            del st.session_state["gmail_user_id"]
    
    @staticmethod
    def add_recent_thread(thread_id: str, session_id: str):
        """Add thread to recent threads (keep last 3)"""
        recent_threads = get_local_storage_item("recent_threads") or []
        
        # Remove if already exists
        recent_threads = [t for t in recent_threads if t["thread_id"] != thread_id]
        
        # Add to beginning
        thread_data = {
            "thread_id": thread_id,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "title": f"Thread {thread_id[-8:]}"  # Short title
        }
        recent_threads.insert(0, thread_data)
        
        # Keep only last 3
        recent_threads = recent_threads[:3]
        
        set_local_storage_item("recent_threads", recent_threads, ttl_hours=24 * 7)  # 1 week TTL
    
    @staticmethod
    def get_recent_threads() -> List[Dict[str, str]]:
        """Get recent threads"""
        return get_local_storage_item("recent_threads") or []
    
    @staticmethod
    def remove_thread_from_recent(thread_id: str):
        """Remove thread from recent threads"""
        recent_threads = get_local_storage_item("recent_threads") or []
        recent_threads = [t for t in recent_threads if t["thread_id"] != thread_id]
        set_local_storage_item("recent_threads", recent_threads, ttl_hours=24 * 7)
    
    @staticmethod
    def clear_all():
        """Clear all localStorage data"""
        remove_local_storage_item("gmail_token")
        remove_local_storage_item("recent_threads")