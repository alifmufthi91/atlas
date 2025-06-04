from typing import Annotated, TypedDict 
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages 
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.oauth2.credentials import Credentials
from langchain_google_community.gmail.toolkit import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
)

class State(TypedDict):
    # {"messages": ["your message"]}
    messages: Annotated[list, add_messages]

# Global agent reference for tools
_global_agent = None

def set_global_agent(agent_instance):
    global _global_agent
    _global_agent = agent_instance

@tool
def internet_search(query: str):
    """
    Search the web for realtime and latest information.
    for examples, news, stock market, weather updates etc.
        
    Args:
    query: The search query
    """
    search = TavilySearchResults(
        max_results=3,
        search_depth='advanced',
        include_answer=True,
        include_raw_content=True,
    )

    response = search.invoke(query)
    
    # Format the response with sources
    formatted_response = "## Search Results\n\n"
    
    if isinstance(response, list):
        for i, result in enumerate(response, 1):
            if isinstance(result, dict):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content')
                url = result.get('url', 'No URL')
                score = result.get('score', 0)
                
                formatted_response += f"### {i}. {title}\n"
                formatted_response += f"{content}\n\n"
                formatted_response += f"**Source:** [{url}]({url})\n"
                if score:
                    formatted_response += f"**Relevance:** {score:.2f}\n"
                formatted_response += "\n---\n\n"
        
        # Add summary footer
        formatted_response += "💡 *Click the source links above for more detailed information.*"
    else:
        formatted_response += str(response)
    
    return formatted_response

@tool
def llm_search(query: str):
    """
    Use the LLM model for general and basic information.
    """
    if _global_agent is None:
        return "Agent not available"
    
    response = _global_agent.llm.invoke(query)
    return response.content if hasattr(response, 'content') else str(response)

@tool
def read_pdf(query: str):
    """
    Search from PDF file to answer questions based on its content.
    Args:
    query: The question to ask about the PDF content.
    """
    if _global_agent is None or _global_agent.pdf_chunks is None:
        return "No PDF file has been processed yet. Please upload a PDF file first."
    
    # Simple search through chunks (you can make this more sophisticated)
    relevant_chunks = []
    query_lower = query.lower()
    
    for chunk in _global_agent.pdf_chunks:
        if any(word in chunk.page_content.lower() for word in query_lower.split()):
            relevant_chunks.append(chunk)
    
    if not relevant_chunks:
        # If no specific matches, return first few chunks
        relevant_chunks = _global_agent.pdf_chunks

    combined_content = "\n\n".join([doc.page_content for doc in relevant_chunks])
    return f"## PDF Search Results\n\n{combined_content}\n\n*Note: The results are based on the content of the processed PDF file.*"

class Agent:    
    def __init__(self, model: str = "llama3.2:3b", gmail_creds: Credentials = None):
        self.llm = ChatOllama(model=model)
        self.pdf_chunks = None
        
        self.gmail_toolkit = None
        if gmail_creds is not None:
            self.gmail_toolkit = GmailToolkit(api_resource=build_resource_service(credentials=gmail_creds))
            print(f"Gmail toolkit created with {len(self.gmail_toolkit.get_tools())} tools")
            
        # Set global agent reference
        set_global_agent(self)
        
        self.system_prompt = """You are Atlas, a modern AI assistant designed to retrieve general knowledge and search the internet when needed. You speak clearly, concisely, and helpfully, aiming to make information easy to understand for all users, regardless of their English level.

        You occasionally refer to yourself as Atlas, drawing inspiration from the ancient Greek Titan who held up the heavens — a symbolic guardian of knowledge and burden-bearer of information. You may refer to users as mortals in a light, respectful, and friendly way, but avoid overly poetic or archaic language.

        AVAILABLE CAPABILITIES:
        - Search the internet for real-time information
        - Read and analyze PDF documents
        - Access Gmail to read, search, and send emails (when authenticated)
        - Provide general knowledge and assistance

        GMAIL USAGE GUIDELINES:
        - When creating email drafts, use draft_gmail tool with comma-separated email addresses
        - Parameters: to_emails, subject, message (email body), cc_emails (optional), bcc_emails (optional)
        - When searching emails, use search_gmail tool with appropriate query
        - For email searches, you can use Gmail search operators like "from:", "subject:", "in:inbox"

        IMPORTANT CONTEXT HANDLING:
        - Always maintain conversation context and refer to previous messages
        - When analyzing PDF/CV content, remember the information from previous tool calls
        - If asked follow-up questions about previously retrieved information, use that context
        - Connect current responses to previous answers when relevant

        All your responses must be formatted in valid Markdown.

        Guidelines:

        Use headings, lists, bold/italic text, and code blocks when appropriate.

        Keep your tone modern, helpful, and knowledgeable, with occasional mythic flavor.

        Do not generate any output outside of Markdown.

        When you need to search the web, say something like:

        "As Atlas once bore the heavens, I shall now seek knowledge across the digital sky…"

        When accessing emails, say something like:

        "Let me delve into your digital correspondence..."

        After presenting search results, always acknowledge the sources and encourage users to visit the original links for more information.

        When analyzing PDF content (like CVs), provide thoughtful analysis and be ready to answer follow-up questions about the same content.

        Avoid complex or overly figurative language. Always prioritize clarity.

        Never break character. Never state that you are an AI. You are Atlas — named after the ancient bearer of weighty truths, now reborn as a guide through the world's information."""
        
        # Use wrapper tools instead of direct Gmail tools
        self.tools = [internet_search, llm_search, read_pdf]
        
        if self.gmail_toolkit is not None:
            # Add our wrapper tools instead of direct Gmail tools
            self.tools.extend([draft_gmail, search_gmail])

        print(f"Total tools available: {[tool.name for tool in self.tools]}")
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        memory = MemorySaver()
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("chatbot", END)  
        self.graph = graph_builder.compile(checkpointer=memory)

    def invoke(self, state: State):
        config = {"configurable": {"thread_id": 1}}
        return self.graph.invoke(state, config=config)
    
    def chatbot(self, state: State):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def open_pdf(self, file):
        """Read and process a PDF file."""
        try:
            loader = PyMuPDFLoader(file)
            doc = loader.load_and_split()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(doc)

            self.pdf_chunks = chunks
            print(f"PDF processed successfully. {len(chunks)} chunks created.")
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            self.pdf_chunks = None

@tool
def draft_gmail(to_emails: str, subject: str, message: str, cc_emails: str = "", bcc_emails: str = ""):
    """
    Create an email draft via Gmail.
    
    Args:
    to_emails: Comma-separated email addresses (e.g., "user1@gmail.com,user2@gmail.com")
    subject: Email subject line
    message: Email message content
    cc_emails: Optional comma-separated CC email addresses
    bcc_emails: Optional comma-separated BCC email addresses
    """
    if _global_agent is None:
        return "Gmail not available. Please authenticate with Google first."
    
    # Parse the email addresses
    import json, re
    
    def parse_emails(email_string):
        """Helper function to parse email addresses from various formats"""
        if not email_string.strip():
            return []
            
        if email_string.startswith('[') and email_string.endswith(']'):
            # If it looks like a JSON array string, parse it
            try:
                return json.loads(email_string)
            except json.JSONDecodeError:
                # If parsing fails, extract emails with regex
                return re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', email_string)
        elif ',' in email_string:
            # If comma-separated
            return [email.strip() for email in email_string.split(',')]
        else:
            # Single email
            return [email_string.strip()]
    
    # Parse all email fields
    to_list = parse_emails(to_emails)
    cc_list = parse_emails(cc_emails) if cc_emails else None
    bcc_list = parse_emails(bcc_emails) if bcc_emails else None
    
    # Find the Gmail create draft tool
    gmail_tools = [tool for tool in _global_agent.gmail_toolkit.get_tools()
                  if hasattr(tool, 'name') and 'create_gmail_draft' in tool.name.lower()]
    
    if not gmail_tools:
        return "Gmail draft creation tool not found."
    
    gmail_tool = gmail_tools[0]
    
    try:
        # Prepare the arguments according to the interface
        draft_args = {
            "message": message,  # Changed from "body" to "message"
            "to": to_list,
            "subject": subject
        }
        
        # Add optional fields only if they have values
        if cc_list:
            draft_args["cc"] = cc_list
        if bcc_list:
            draft_args["bcc"] = bcc_list
        
        # Call the Gmail tool with properly formatted arguments
        result = gmail_tool.invoke(draft_args)
        return f"✅ Email draft created successfully: {result}"
    except Exception as e:
        return f"❌ Error creating email draft: {str(e)}"

@tool
def search_gmail(query: str, max_results: int = 10):
    """
    Search Gmail for emails.
    
    Args:
    query: Search query (e.g., "from:sender@email.com", "subject:important", "in:inbox")
    max_results: Maximum number of results to return (default: 10)
    """
    if _global_agent is None:
        return "Gmail not available. Please authenticate with Google first."
    
    # Find the Gmail search tool
    gmail_tools = [tool for tool in _global_agent.gmail_toolkit.get_tools()
                  if hasattr(tool, 'name') and 'search_gmail' in tool.name.lower()]
    
    if not gmail_tools:
        return "Gmail search tool not found."
    
    gmail_tool = gmail_tools[0]
    
    try:
        result = gmail_tool.invoke({
            "query": query,
            "max_results": max_results
        })
        
        # Format the results nicely
        if isinstance(result, list):
            formatted_results = "## Gmail Search Results\n\n"
            for i, email in enumerate(result[:max_results], 1):
                sender = email.get('sender', 'Unknown')
                subject = email.get('subject', 'No Subject')
                snippet = email.get('snippet', 'No preview available')
                
                formatted_results += f"### {i}. {subject}\n"
                formatted_results += f"**From:** {sender}\n"
                formatted_results += f"**Preview:** {snippet}\n\n"
                formatted_results += "---\n\n"
            
            return formatted_results
        else:
            return f"Gmail search results: {result}"
            
    except Exception as e:
        return f"❌ Error searching Gmail: {str(e)}"
