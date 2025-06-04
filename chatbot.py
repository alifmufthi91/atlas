import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
import re
from agent import Agent
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
import os
import json
from google.oauth2 import id_token
from google.auth.transport import requests
from authlib.integrations.requests_client import OAuth2Session
from urllib.parse import urlencode, urlparse, parse_qs
from langchain_google_community.gmail.toolkit import GmailToolkit
from google.oauth2.credentials import Credentials


def process_ollama_response(ollama_output):
    """
    Parses Ollama output, separates thinking sections and response,
    and formats thinking sections as code markdown.
    """
    print(ollama_output)
    # Regex to capture content within <think> tags and the actual response
    match = re.match(r"(<think>(.*?)</think>)?(.*)", ollama_output, re.DOTALL)
    
    think_content = match.group(2) if match.group(1) else None
    response_content = match.group(3).strip()  # Remove leading/trailing whitespace

    return think_content, response_content

@st.cache_resource
def get_agent_without_gmail():
    """Create agent without Gmail credentials (cached)"""
    return Agent()

def get_agent_with_gmail(gmail_credentials):
    """Create agent with Gmail credentials (not cached due to unhashable credentials)"""
    return Agent(gmail_creds=gmail_credentials)

@st.cache_resource
def get_google_credential():
    credentials_path = os.path.join(os.path.dirname(__file__), 'credentials.json')
    with open(credentials_path, 'r') as f:
        client_secrets = json.load(f)
    return client_secrets

def main():
    load_dotenv('.env')
    st.set_page_config(page_title="Atlas AI", page_icon="🤖")
    st.title("🤖 Atlas AI")
    
    SCOPES = [
        "openid",
        "email", 
        "profile",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send"
    ]
    
    # Handle OAuth callback first
    query_params = st.query_params.to_dict()
    if "code" in query_params and "gmail_credentials" not in st.session_state:
        try:
            code = query_params["code"]
            
            # Create session and fetch token
            session = OAuth2Session(
                get_google_credential()['web']['client_id'], 
                get_google_credential()['web']['client_secret'], 
                redirect_uri=get_google_credential()['web']['redirect_uris'][0]
            )
            token = session.fetch_token(get_google_credential()['web']['token_uri'], code=code)

            # Save credentials in session
            st.session_state["gmail_credentials"] = Credentials(
                token=token["access_token"],
                refresh_token=token.get("refresh_token"),
                token_uri=get_google_credential()['web']['token_uri'],
                client_id=get_google_credential()['web']['client_id'],
                client_secret=get_google_credential()['web']['client_secret'],
                scopes=SCOPES
            )
            
            st.success("✅ Successfully authenticated with Google!")
            
            # Clear the code from URL and rerun
            st.query_params.clear()
            st.rerun()
            
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
    
    with st.sidebar:
        # Check if user is already authenticated
        st.write("**Gmail Authentication**")
        if "gmail_credentials" not in st.session_state:
            
            # Simple redirect-based OAuth
            creds = get_google_credential()
            query_params = {
                "client_id": creds['web']['client_id'],
                "response_type": "code",
                "scope": " ".join(SCOPES),
                "redirect_uri": creds['web']['redirect_uris'][0],
                "access_type": "offline",
                "prompt": "consent"
            }
            
            auth_url = f"{creds['web']['auth_uri']}?{urlencode(query_params)}"
            
            st.markdown(f"""
            <a href="{auth_url}" target="_self" style="
                display: inline-block;
                background: #4285f4;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                font-size: 14px;
            ">🔐 Login with Google</a>
            """, unsafe_allow_html=True)
            
        else:
            if st.button("Logout"):
                # Clear Gmail credentials
                if "gmail_credentials" in st.session_state:
                    del st.session_state["gmail_credentials"]
                st.rerun()
    
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        
    if "pdf_loaded" not in st.session_state:
        st.session_state["pdf_loaded"] = False

    # Create agent
    gmail_creds = st.session_state.get("gmail_credentials")
    if gmail_creds:
        agent = get_agent_with_gmail(gmail_creds)
    else:
        agent = get_agent_without_gmail()
    
    if uploaded_file is not None and not st.session_state["pdf_loaded"]:
        # Write the bytesIO object to temp file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        # Load the PDF file
        agent.open_pdf("temp.pdf")
        # remove the temp file
        os.remove("temp.pdf")
        # add system message that the agent has loaded the PDF
        st.session_state["pdf_loaded"] = True
        st.session_state["messages"].append({
            "role": "system",
            "content": "The PDF file has been loaded successfully. You can now ask questions about the content."
        })

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message(msg["role"]).markdown(f'''{msg["content"]}''')

    if prompt := st.chat_input("Enter a prompt"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking...", show_time=True):
                response = agent.invoke({"messages": st.session_state.messages})
                print(response)
                msg = response['messages'][-1].content
                thinking, answer = process_ollama_response(msg)
            st.markdown(f'''{answer}''')

        # put the code block thinking and answer to session state
        st.session_state.messages.append({"role": "assistant", "content": answer, "thinking": thinking})


if __name__ == "__main__":
    main()