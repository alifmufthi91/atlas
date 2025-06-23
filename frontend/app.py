import streamlit as st
import requests
import json
from typing import List, Dict, Any
import base64
from backend_client import BackendClient
from storage_utils import LocalStorageManager

# Backend configuration
BACKEND_URL = "http://localhost:8000"

def validate_and_clean_tokens(client):
    """Validate stored tokens with backend and clean invalid ones"""
    tokens = {}
    gmail_token = st.session_state.get("gmail_access_token")
    if gmail_token:
        tokens["gmail"] = gmail_token
    
    if tokens:
        try:
            validation = client.validate_tokens(tokens)
            
            # Remove invalid tokens from storage
            if validation["needs_reauth"]:
                for provider in validation["invalid_tokens"]:
                    if provider == "gmail":
                        LocalStorageManager.remove_gmail_token()
                
                # Return only valid tokens
                return validation["valid_tokens"]
            
            return tokens
            
        except requests.exceptions.RequestException as e:
            st.warning("⚠️ Could not validate tokens with backend")
            return {}
    
    return {}

def check_backend_health(client):
    """Check if backend is accessible"""
    try:
        response = client.session.get(f"{BACKEND_URL}/")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def main():
    st.set_page_config(page_title="Atlas AI", page_icon="🤖")
    st.title("🤖 Atlas AI")
    
    # Initialize backend client
    if "backend_client" not in st.session_state:
        st.session_state.backend_client = BackendClient(BACKEND_URL)
    
    client = st.session_state.backend_client
    
    # Check backend connectivity first
    if not check_backend_health(client):
        st.error("🔌 Cannot connect to backend. Please check if it's running on http://localhost:8000")
        st.stop()
    
    # Try to restore Gmail token from storage on startup
    if "gmail_access_token" not in st.session_state:
        LocalStorageManager.get_gmail_token()
    
    # Validate tokens with backend and clean invalid ones
    valid_tokens = validate_and_clean_tokens(client)
    
    # Handle OAuth callbacks
    query_params = st.query_params.to_dict()
    auth_code = query_params.get("code")
    auth_state = query_params.get("state")
    
    if auth_code and auth_state == "gmail" and "gmail_access_token" not in st.session_state:
        try:
            with st.spinner("Authenticating with Gmail..."):
                auth_response = client.authenticate_with_google(auth_code)
                
                # Store with 1 day TTL
                LocalStorageManager.set_gmail_token(
                    auth_response["access_token"], 
                    auth_response["user_id"]
                )
                
            st.query_params.clear()
            st.rerun()
            
        except requests.exceptions.RequestException as e:
            st.error(f"Gmail authentication failed: {e}")
    
    # Use validated tokens for session ID
    session_id = "_".join(sorted(valid_tokens.keys())) if valid_tokens else "anonymous"
    
    # Initialize thread management
    if "current_thread_id" not in st.session_state:
        recent_threads = LocalStorageManager.get_recent_threads()
        if recent_threads and recent_threads[0]["session_id"] == session_id:
            st.session_state.current_thread_id = recent_threads[0]["thread_id"]
        else:
            st.session_state.current_thread_id = f"thread_{session_id}_main"
            # Add this initial thread to recent threads
            LocalStorageManager.add_recent_thread(st.session_state.current_thread_id, session_id, "Main Conversation")
    
    # Sidebar
    with st.sidebar:
        st.write("**Available Integrations**")
        
        # Gmail Integration - use valid_tokens instead of tokens
        gmail_token = valid_tokens.get("gmail")
        if not gmail_token:
            if st.button("🔗 Connect Gmail"):
                try:
                    auth_url = client.get_google_auth_url()
                    st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', 
                              unsafe_allow_html=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to get Gmail auth URL: {e}")
        else:
            if st.button("Disconnect Gmail"):
                LocalStorageManager.remove_gmail_token()
                st.rerun()
        
        st.divider()
        
        # Thread Management
        st.write("**Conversation Threads**")
        
        # Current thread info
        current_thread = st.session_state.current_thread_id
        st.write(f"**Current:** `{current_thread[-8:]}`")
        
        # New thread button
        if st.button("🆕 New Conversation"):
            import uuid
            new_thread_id = f"thread_{uuid.uuid4().hex[:8]}"
            st.session_state.current_thread_id = new_thread_id
            
            # Add to recent threads
            LocalStorageManager.add_recent_thread(new_thread_id, session_id, "New Conversation")
            
            # Reset messages
            thread_key = f"messages_{new_thread_id}"
            st.session_state[thread_key] = [{"role": "assistant", "content": "How can I help you?"}]
            st.rerun()
        
        # Show recent threads
        recent_threads = LocalStorageManager.get_recent_threads()
        if recent_threads:
            st.write("**Recent Conversations:**")
            for thread_data in recent_threads:
                thread_id = thread_data["thread_id"]
                is_current = thread_id == st.session_state.current_thread_id
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    button_text = f"{'🔹' if is_current else '💬'} {thread_data['title']}"
                    if st.button(button_text, key=f"thread_{thread_id}", disabled=is_current):
                        st.session_state.current_thread_id = thread_id
                        
                        # Update last used timestamp
                        LocalStorageManager.update_thread_last_used(thread_id)
                        
                        # Initialize messages for this thread if not exists
                        thread_key = f"messages_{thread_id}"
                        if thread_key not in st.session_state:
                            st.session_state[thread_key] = [{"role": "assistant", "content": "Continuing previous conversation..."}]
                        
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{thread_id}"):
                        LocalStorageManager.remove_thread_from_recent(thread_id)
                        
                        if thread_id == st.session_state.current_thread_id:
                            import uuid
                            new_thread_id = f"thread_{uuid.uuid4().hex[:8]}"
                            st.session_state.current_thread_id = new_thread_id
                            LocalStorageManager.add_recent_thread(new_thread_id, session_id, "New Conversation")
                            
                            thread_key = f"messages_{new_thread_id}"
                            st.session_state[thread_key] = [{"role": "assistant", "content": "How can I help you?"}]
                        
                        st.rerun()
        
        # Show available tools
        st.divider()
        st.write("**Available Tools:**")
        st.write("🤖 Basic LLM Chat")
        st.write("🌐 Internet Search")
        st.write("📄 PDF Analysis")
        
        if gmail_token:
            st.write("📧 Gmail Tools")
            
        # Debug section with backend validation
        with st.expander("🔧 Debug Storage & Backend"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear All Storage"):
                    LocalStorageManager.clear_all()
                    st.success("Cleared all storage data")
                    st.rerun()
            
            with col2:
                if st.button("Validate Tokens"):
                    validated = validate_and_clean_tokens(client)
                    if validated:
                        st.success(f"✅ Valid: {list(validated.keys())}")
                    else:
                        st.info("No valid tokens")
            
            # Show detailed storage status
            status = LocalStorageManager.get_storage_status()
            
            st.write("**Storage Status:**")
            st.json(status)
            
            st.write("**Backend Status:**")
            backend_healthy = check_backend_health(client)
            st.write(f"Backend: {'✅ Connected' if backend_healthy else '❌ Disconnected'}")
            
            st.write("**Current Session:**")
            st.write(f"Session ID: `{session_id}`")
            st.write(f"Valid tokens: {list(valid_tokens.keys())}")
    
    # Main content
    
    # PDF Upload
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    
    # Initialize messages per thread
    thread_key = f"messages_{st.session_state.current_thread_id}"
    if thread_key not in st.session_state:
        st.session_state[thread_key] = [{"role": "assistant", "content": "How can I help you?"}]
    
    current_messages = st.session_state[thread_key]
    
    # PDF processing
    if uploaded_file is not None:
        pdf_key = f"pdf_loaded_{st.session_state.current_thread_id}"
        if pdf_key not in st.session_state:
            try:
                with st.spinner("Processing PDF..."):
                    file_content = uploaded_file.read()
                    result = client.upload_pdf_session(file_content, uploaded_file.name, session_id)
                    
                if result["success"]:
                    st.session_state[pdf_key] = True
                    current_messages.append({
                        "role": "system",
                        "content": result["message"]
                    })
                    st.session_state[thread_key] = current_messages
                    st.success(result["message"])
                    
            except requests.exceptions.RequestException as e:
                st.error(f"PDF upload failed: {e}")
    
    # Display chat messages
    for msg in current_messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message(msg["role"]).markdown(msg["content"])
    
    st.empty()
    # Chat input
    if prompt := st.chat_input("Enter a prompt"):
        # Add user message to current thread
        current_messages.append({"role": "user", "content": prompt})
        st.session_state[thread_key] = current_messages
        
        # Update recent threads (mark as recently used)
        LocalStorageManager.add_recent_thread(st.session_state.current_thread_id, session_id)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Send only the new user message
                    new_message = [{"role": "user", "content": prompt}]
                    
                    response = client.chat(
                        messages=new_message,
                        session_id=session_id,
                        thread_id=st.session_state.current_thread_id,
                        auth_tokens=valid_tokens  # Use validated tokens
                    )
                    
                    content = response["content"]
                    thinking = response.get("thinking")
                    
                    st.markdown(content)
                    
                    # Add assistant response to current thread
                    current_messages.append({
                        "role": "assistant", 
                        "content": content,
                        "thinking": thinking
                    })
                    st.session_state[thread_key] = current_messages
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Chat request failed: {e}")
                    # If it's a token validation error, suggest refreshing
                    if "token" in str(e).lower():
                        st.info("💡 Try clicking 'Validate Tokens' in the debug section")

if __name__ == "__main__":
    main()