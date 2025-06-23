# 🤖 Atlas AI - Intelligent Document & Communication Assistant

Atlas AI is a modern AI assistant powered by LangGraph and multiple LLM backends, designed to help you interact with documents, search the web, and manage your communications seamlessly.

## ✨ Features

### 🧠 **Core AI Capabilities**
- **Multi-LLM Support**: Ollama (local) and OpenAI-compatible models
- **Intelligent Query Classification**: Automatically routes queries to appropriate tools
- **Thread-based Conversations**: Maintain context across multiple conversation threads
- **Persistent Storage**: Conversations and settings persist across sessions

### 📄 **Document Processing**
- **PDF Analysis**: Upload and analyze PDF documents with semantic chunking
- **Content Search**: Ask questions about uploaded document content
- **Context-aware Responses**: Get answers based on document content

### 🌐 **Web Integration**
- **Real-time Search**: Access current news, stock prices, weather, and trending topics
- **Sourced Information**: Get clickable links for verification
- **Advanced Search**: Powered by Tavily Search for comprehensive results

### 📧 **Gmail Integration**
- **Email Search**: Use Gmail search operators (`from:`, `subject:`, `in:`, etc.)
- **Draft Creation**: Compose emails with TO, CC, and BCC support
- **OAuth Authentication**: Secure Google account integration

### 📅 **Google Calendar Integration**
- **Event Search**: Find events by time range (today, week, month)
- **Calendar Management**: Access multiple calendars
- **Smart Scheduling**: Get formatted event information

## 🏗️ Architecture

```
Atlas AI
├── frontend/          # Streamlit web interface
│   ├── app.py         # Main application
│   ├── backend_client.py    # API client
│   └── storage_utils.py     # Persistent storage
├── backend/           # FastAPI server
│   ├── main.py        # API endpoints
│   └── agent.py       # LangGraph agent logic
└── credentials.json   # Google OAuth config
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama (for local LLM)
- Google Cloud Project (for Gmail/Calendar integration)

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma3:12b  # or your preferred model
```

### 2. Configure Google OAuth (Optional)

1. Create a Google Cloud Project
2. Enable Gmail and Calendar APIs
3. Create OAuth 2.0 credentials
4. Download and save as `credentials.json`

### 3. Set Environment Variables

Create a `.env` file in both `frontend/` and `backend/` directories:

```bash
# Tavily API for web search
TAVILY_API_KEY=your_tavily_api_key

# OpenAI-compatible API (optional)
OPENAI_API_KEY=your_api_key
```

### 4. Start the Application

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
streamlit run app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 🎯 Usage Examples

### Basic Chat
```
User: Hello, how are you?
Atlas: Hello! I'm Atlas, your AI assistant. I'm doing well and ready to help you with any questions or tasks!
```

### Document Analysis
1. Upload a PDF file
2. Ask questions about the content:
```
User: What are the main topics covered in this document?
Atlas: Based on the uploaded PDF, the main topics include...
```

### Web Search
```
User: What's the latest news about AI?
Atlas: ## Search Results

### 1. Latest AI Breakthroughs in 2024
Recent developments in artificial intelligence...
**Source:** [example.com](https://example.com)
```

### Gmail Integration
```
User: Search for emails from john@company.com
Atlas: ## Gmail Search Results

### 1. Project Update
**From:** john@company.com
**Preview:** Here's the latest update on our project...
```

### Calendar Events
```
User: What's on my calendar this week?
Atlas: ## 📅 Your Calendar Events
**Time Range:** Week (2024-01-15 to 2024-01-22)

### 1. Team Meeting
📅 **Calendar:** Work Calendar
⏰ **Start:** Monday, January 15, 2024 at 09:00 AM
```

## 🔧 Configuration

### Model Configuration

Edit `backend/agent.py` to change the default models:

```python
def __init__(self, model: str = "gemma3:12b", google_creds: Credentials = None):
    self.llm_local = ChatOllama(model=model)  # Change model here
    self.llm_open_ai = ChatOpenAI(
        model="gemini-2.5-flash",  # Or gpt-4, claude-3, etc.
        api_key="your_api_key",
        base_url="https://api.openai.com/v1/"
    )
```

### Storage Configuration

The application uses file-based storage in `frontend/.streamlit_storage`. To clear all data:

```bash
rm -rf frontend/.streamlit_storage/
```

## 🛠️ Development

### Project Structure

```
├── frontend/
│   ├── app.py              # Main Streamlit app
│   ├── backend_client.py   # API communication
│   ├── storage_utils.py    # Data persistence
│   └── local_storage_utils.py  # Browser storage (alternative)
├── backend/
│   ├── main.py            # FastAPI server
│   ├── agent.py           # LangGraph agent
│   └── credentials.json   # Google OAuth config
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Key Components

- **LangGraph Agent**: `backend/agent.py` - Core AI logic with tool routing
- **FastAPI Backend**: `backend/main.py` - REST API endpoints
- **Streamlit Frontend**: `frontend/app.py` - Web interface
- **Storage System**: `frontend/storage_utils.py` - Persistent data

### Adding New Tools

1. Define tool function in `backend/agent.py`:
```python
@tool
def my_custom_tool(query: str):
    """Description of your tool."""
    # Your implementation
    return result
```

2. Add tool to the agent's tool list:
```python
self.tools = [internet_search, llm_search, read_pdf, my_custom_tool]
```

## 🔐 Security Notes

- **Credentials**: Never commit `credentials.json` to version control
- **API Keys**: Store sensitive keys in `.env` files
- **OAuth Tokens**: Tokens are stored locally with TTL expiration
- **Backend**: Run backend on localhost for development only

## 🐛 Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure backend is running on port 8000
   - Check firewall settings

2. **Ollama Model Not Found**
   ```bash
   ollama list  # Check available models
   ollama pull gemma3:12b  # Download model
   ```

3. **Google Auth Fails**
   - Verify `credentials.json` format
   - Check redirect URIs in Google Console
   - Ensure APIs are enabled

4. **PDF Upload Issues**
   - Check file size limits
   - Ensure PDF is not password-protected

### Debug Tools

Use the debug section in the sidebar to:
- Check storage status
- Validate tokens
- Clear all data
- View backend connectivity

## 📝 License

This project is open source. Feel free to modify and distribute according to your needs.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🌟 Acknowledgments

- **LangGraph**: For the agent framework
- **Streamlit**: For the web interface
- **Ollama**: For local LLM support
- **FastAPI**: For the backend API
- **Google APIs**: For Gmail and Calendar integration

---

**Atlas AI** - Making knowledge accessible, one conversation at a time. 🚀