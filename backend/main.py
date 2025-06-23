from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv
import json
from google.oauth2.credentials import Credentials
from authlib.integrations.requests_client import OAuth2Session
from urllib.parse import urlencode
import jwt
from datetime import datetime, timedelta
import secrets

# Import your agent
from agent import Agent

load_dotenv()

app = FastAPI(title="Atlas AI Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (use Redis/Database in production)
user_sessions = {}  # session_id -> {agents, credentials, pdf_chunks}
provider_tokens = {}  # token -> credentials mapping

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: str
    thread_id: Optional[str] = None  # Add thread_id
    auth_tokens: Optional[Dict[str, str]] = {}

class ChatResponse(BaseModel):
    content: str
    thinking: Optional[str] = None

class AuthRequest(BaseModel):
    code: str

class AuthResponse(BaseModel):
    access_token: str
    user_id: str

class PDFUploadRequest(BaseModel):
    file_content: str  # base64 encoded
    filename: str

class PDFUploadResponse(BaseModel):
    success: bool
    message: str

def get_google_credentials():
    """Load Google OAuth credentials"""
    credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
    with open(credentials_path, 'r') as f:
        return json.load(f)

@app.get("/")
async def root():
    return {"message": "Atlas AI Backend API"}

@app.get("/auth/google-url")
async def get_google_auth_url(state: str = "gmail"):
    """Get Google OAuth URL with state parameter"""
    creds = get_google_credentials()
    
    SCOPES = [
        "openid",
        "email", 
        "profile",
        "https://mail.google.com/",
        "https://www.googleapis.com/auth/calendar"
    ]
    
    query_params = {
        "client_id": creds['web']['client_id'],
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "redirect_uri": creds['web']['redirect_uris'][0],
        "access_type": "offline",
        "prompt": "consent",
        "state": state
    }
    
    auth_url = f"{creds['web']['auth_uri']}?{urlencode(query_params)}"
    return {"auth_url": auth_url}

@app.post("/auth/google-callback", response_model=AuthResponse)
async def google_callback(auth_request: AuthRequest):
    """Handle Google OAuth callback"""
    try:
        creds = get_google_credentials()
        
        session = OAuth2Session(
            creds['web']['client_id'], 
            creds['web']['client_secret'], 
            redirect_uri=creds['web']['redirect_uris'][0]
        )
        token = session.fetch_token(creds['web']['token_uri'], code=auth_request.code)
        
        gmail_creds = Credentials(
            token=token["access_token"],
            refresh_token=token.get("refresh_token"),
            token_uri=creds['web']['token_uri'],
            client_id=creds['web']['client_id'],
            client_secret=creds['web']['client_secret'],
            scopes=[
                "openid", "email", "profile",
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send"
            ]
        )
        
        user_id = f"gmail_user_{secrets.token_hex(8)}"
        access_token = f"gmail_token_{secrets.token_hex(16)}"
        
        # Store the credentials with the token for later retrieval
        provider_tokens[access_token] = {
            "provider": "gmail",
            "credentials": gmail_creds,
            "user_id": user_id,
            "created_at": datetime.utcnow()
        }
        
        return AuthResponse(access_token=access_token, user_id=user_id)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - works with or without authentication"""
    try:
        session_id = request.session_id
        thread_id = request.thread_id or f"thread_{session_id}"
        auth_tokens = request.auth_tokens
        
        print(f"🔄 Processing chat for session: {session_id}, thread: {thread_id}")
        
        # Get or create session
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                "agents": {},
                "credentials": {},
                "pdf_chunks": None,
                "threads": {}
            }
        
        session = user_sessions[session_id]
        
        # Track thread usage
        if thread_id not in session["threads"]:
            session["threads"][thread_id] = {
                "created_at": datetime.utcnow(),
                "message_count": 0
            }
        
        session["threads"][thread_id]["message_count"] += 1
        session["threads"][thread_id]["last_used"] = datetime.utcnow()
        
        # Retrieve actual credentials from tokens
        current_credentials = {}
        for provider, token in auth_tokens.items():
            if token in provider_tokens:
                token_data = provider_tokens[token]
                if token_data["provider"] == provider:
                    current_credentials[provider] = token_data["credentials"]
                    print(f"✅ Retrieved {provider} credentials for session {session_id}")
                else:
                    print(f"⚠️  Token provider mismatch: expected {provider}, got {token_data['provider']}")
            else:
                print(f"❌ Token not found for {provider}: {token}")
        
        # Create appropriate agent key based on available credentials
        agent_key = "_".join(sorted(current_credentials.keys())) if current_credentials else "basic"
        
        print(f"🎯 Target agent key: '{agent_key}'")
        print(f"🎯 Available credentials: {list(current_credentials.keys())}")
        print(f"🎯 Existing agents: {list(session['agents'].keys())}")
        
        # Check if credentials have changed since last time
        stored_credentials = session.get("credentials", {})
        credentials_changed = set(current_credentials.keys()) != set(stored_credentials.keys())
        
        if credentials_changed:
            print(f"🔄 Credentials changed from {list(stored_credentials.keys())} to {list(current_credentials.keys())}")
            # Clear all existing agents when credentials change
            session["agents"].clear()
            print("🗑️  Cleared all existing agents due to credential change")
        
        # Create or get agent
        if agent_key not in session["agents"]:
            # Create agent with available credentials
            gmail_creds = current_credentials.get("gmail")
            
            print(f"🤖 Creating NEW agent '{agent_key}' with Gmail: {gmail_creds is not None}")
            
            if gmail_creds:
                print(f"📧 Gmail credentials details:")
                print(f"   - Token: {gmail_creds.token[:20]}...")
                print(f"   - Has refresh token: {gmail_creds.refresh_token is not None}")
                print(f"   - Scopes: {gmail_creds.scopes}")
            
            session["agents"][agent_key] = Agent(google_creds=gmail_creds)
            
            # Store credentials in session for future reference
            session["credentials"] = current_credentials
            
            # Restore PDF chunks to the new agent
            if session["pdf_chunks"]:
                session["agents"][agent_key].pdf_chunks = session["pdf_chunks"]
                print(f"📄 Restored {len(session['pdf_chunks'])} PDF chunks to agent '{agent_key}'")
            else:
                print(f"📄 No PDF chunks to restore for agent '{agent_key}'")
        else:
            print(f"♻️  Using existing agent '{agent_key}'")
        
        agent = session["agents"][agent_key]
        
        # Debug: Check agent state in detail
        print(f"🔍 Agent '{agent_key}' detailed state:")
        print(f"   - Has PDF chunks: {agent.pdf_chunks is not None}")
        if agent.pdf_chunks:
            print(f"   - PDF chunks count: {len(agent.pdf_chunks)}")
        
        print(f"   - Has gmail_toolkit: {hasattr(agent, 'gmail_toolkit')}")
        if hasattr(agent, 'gmail_toolkit'):
            print(f"   - Gmail toolkit is None: {agent.gmail_toolkit is None}")
            if agent.gmail_toolkit:
                gmail_tools = agent.gmail_toolkit.get_tools()
                gmail_tool_names = [tool.name for tool in gmail_tools]
                print(f"   - Gmail tools count: {len(gmail_tools)}")
                print(f"   - Gmail tool names: {gmail_tool_names}")
            else:
                print("   - Gmail toolkit is None (no Gmail credentials)")
        else:
            print("   - No gmail_toolkit attribute")
        
        # Double-check agent tools
        if hasattr(agent, 'tools'):
            tool_names = [tool.name for tool in agent.tools]
            print(f"   - All agent tools: {tool_names}")
            gmail_tool_count = len([t for t in tool_names if 'gmail' in t.lower()])
            print(f"   - Gmail tools in agent.tools: {gmail_tool_count}")
        
        # Convert messages to LangChain format (only send the current message)
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Only send the last user message to avoid duplication
        if messages and messages[-1]["role"] == "user":
            current_user_message = messages[-1]
            state = {
                "messages": [current_user_message],
                "thread_id": thread_id
            }
        else:
            state = {
                "messages": messages,
                "thread_id": thread_id
            }
        
        print(f"💬 Sending {len(state['messages'])} message(s) to agent")
        
        # Get response from agent with thread_id
        response = agent.invoke(state, thread_id=thread_id)
        content = response['messages'][-1].content
        
        # Process response
        import re
        match = re.match(r"(<think>(.*?)</think>)?(.*)", content, re.DOTALL)
        thinking = match.group(2) if match.group(1) else None
        answer = match.group(3).strip() if match.group(3) else content
        
        print(f"✅ Chat completed for thread {thread_id}")
        
        return ChatResponse(content=answer, thinking=thinking)
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# Add endpoints to manage threads
@app.get("/sessions/{session_id}/threads")
async def get_session_threads(session_id: str):
    """Get all threads for a session"""
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    threads = user_sessions[session_id].get("threads", {})
    return {
        "session_id": session_id,
        "threads": {
            thread_id: {
                "created_at": data["created_at"].isoformat(),
                "last_used": data.get("last_used", data["created_at"]).isoformat(),
                "message_count": data["message_count"]
            }
            for thread_id, data in threads.items()
        }
    }

@app.delete("/sessions/{session_id}/threads/{thread_id}")
async def delete_thread(session_id: str, thread_id: str):
    """Delete a specific thread"""
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = user_sessions[session_id]
    if thread_id in session.get("threads", {}):
        del session["threads"][thread_id]
        print(f"🗑️  Deleted thread {thread_id} from session {session_id}")
        return {"message": f"Thread {thread_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Thread not found")

@app.post("/sessions/{session_id}/threads/new")
async def create_new_thread(session_id: str):
    """Create a new thread for a session"""
    import uuid
    
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "agents": {},
            "credentials": {},
            "pdf_chunks": None,
            "threads": {}
        }
    
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    user_sessions[session_id]["threads"][thread_id] = {
        "created_at": datetime.utcnow(),
        "message_count": 0
    }
    
    return {"thread_id": thread_id, "session_id": session_id}

@app.post("/upload-pdf-anonymous", response_model=PDFUploadResponse)
async def upload_pdf_anonymous(request: PDFUploadRequest):
    """Upload and process PDF without authentication"""
    try:
        # Decode base64 file content
        import base64
        file_content = base64.b64decode(request.file_content)
        
        # Use anonymous session
        session_id = "anonymous"
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                "agents": {},
                "credentials": {},
                "pdf_chunks": None
            }
        
        session = user_sessions[session_id]
        
        # Create basic agent if not exists
        if "basic" not in session["agents"]:
            session["agents"]["basic"] = Agent()
        
        agent = session["agents"]["basic"]
        
        # Save temporary file
        temp_path = f"temp_anonymous_{request.filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # Process PDF
        agent.open_pdf(temp_path)
        
        # Store PDF chunks in session for other agents
        session["pdf_chunks"] = agent.pdf_chunks
        
        # Clean up
        os.remove(temp_path)
        
        return PDFUploadResponse(
            success=True, 
            message=f"PDF '{request.filename}' processed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

@app.delete("/logout/{provider}")
async def logout_provider(provider: str, token: str = None):
    """Logout from specific provider"""
    if token and token in provider_tokens:
        del provider_tokens[token]
        print(f"🔓 Logged out {provider} token: {token}")
    
    # Clean up sessions that use this provider
    for session_id, session_data in user_sessions.items():
        if provider in session_data.get("credentials", {}):
            # Remove the provider from credentials
            del session_data["credentials"][provider]
            
            # IMPORTANT: Clear ALL agents to force recreation with new credentials
            old_agents = list(session_data["agents"].keys())
            session_data["agents"].clear()
            
            print(f"🗑️  Cleared all agents in session {session_id} after {provider} logout: {old_agents}")
    
    return {"message": f"Logged out from {provider} successfully"}

# Add a debug endpoint to check token status
@app.get("/debug/tokens")
async def debug_tokens():
    """Debug endpoint to check stored tokens"""
    return {
        "active_tokens": len(provider_tokens),
        "tokens": {
            token: {
                "provider": data["provider"],
                "user_id": data["user_id"],
                "has_credentials": data["credentials"] is not None,
                "created_at": data["created_at"].isoformat()
            }
            for token, data in provider_tokens.items()
        }
    }

@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to check session status"""
    return {
        "active_sessions": len(user_sessions),
        "sessions": {
            session_id: {
                "agents": list(session_data["agents"].keys()),
                "credentials": list(session_data.get("credentials", {}).keys()),
                "has_pdf": session_data.get("pdf_chunks") is not None
            }
            for session_id, session_data in user_sessions.items()
        }
    }

@app.post("/sessions/{session_id}/refresh-agents")
async def refresh_agents(session_id: str):
    """Force recreation of all agents in a session"""
    if session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = user_sessions[session_id]
    
    # Clear all agents
    old_agents = list(session["agents"].keys())
    session["agents"].clear()
    
    print(f"🔄 Cleared agents for session {session_id}: {old_agents}")
    
    return {
        "message": f"Refreshed agents for session {session_id}",
        "cleared_agents": old_agents
    }

@app.post("/validate-tokens")
async def validate_tokens(request: Dict[str, str]):
    """Validate if tokens are still valid in backend storage"""
    valid_tokens = {}
    invalid_tokens = []
    
    for provider, token in request.items():
        if token in provider_tokens:
            valid_tokens[provider] = token
            print(f"✅ Valid token for {provider}")
        else:
            invalid_tokens.append(provider)
            print(f"❌ Invalid token for {provider}")
    
    return {
        "valid_tokens": valid_tokens,
        "invalid_tokens": invalid_tokens,
        "needs_reauth": len(invalid_tokens) > 0
    }