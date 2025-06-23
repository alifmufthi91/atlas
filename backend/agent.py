from typing import Annotated, TypedDict, Literal, Optional
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages 
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.oauth2.credentials import Credentials
from langchain_google_community.gmail.toolkit import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service as build_gmail_resource_service,
)
from langchain_google_community.calendar.utils import (
    build_resource_service as build_calendar_resource_service,
)
from langchain_google_community import CalendarToolkit

class State(TypedDict):
    thread_id: str  # Thread ID for conversation context
    messages: Annotated[list, add_messages]
    query_type: str  # Add query classification
    needs_tools: bool  # Whether tools are needed

# Global agent reference for tools
_global_agent = None

def set_global_agent(agent_instance):
    global _global_agent
    _global_agent = agent_instance

# Keep your existing tools (internet_search, llm_search, read_pdf, draft_gmail, search_gmail)
@tool
def internet_search(query: str):
    """Search the web for realtime and latest information."""
    search = TavilySearchResults(
        max_results=3,
        search_depth='advanced',
        include_answer=True,
        include_raw_content=True,
    )

    response = search.invoke(query)
    
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
        
        formatted_response += "💡 *Click the source links above for more detailed information.*"
    else:
        formatted_response += str(response)
    
    return formatted_response

@tool
def llm_search(query: str):
    """Use the LLM model for general and basic information."""
    if _global_agent is None:
        return "Agent not available"
    
    response = _global_agent.llm.invoke(query)
    return response.content if hasattr(response, 'content') else str(response)

@tool
def read_pdf(query: str):
    """Search from PDF file to answer questions based on its content."""
    if _global_agent is None or _global_agent.pdf_chunks is None:
        return "No PDF file has been processed yet. Please upload a PDF file first."
    
    relevant_chunks = []
    query_lower = query.lower()
    
    for chunk in _global_agent.pdf_chunks:
        if any(word in chunk.page_content.lower() for word in query_lower.split()):
            relevant_chunks.append(chunk)
    
    if not relevant_chunks:
        relevant_chunks = _global_agent.pdf_chunks[:3]  # Return first 3 chunks

    combined_content = "\n\n".join([doc.page_content for doc in relevant_chunks])
    return f"## PDF Search Results\n\n{combined_content}\n\n*Note: The results are based on the content of the processed PDF file.*"


@tool
def draft_gmail(to_emails: str, subject: str, message: str, cc_emails: str = "", bcc_emails: str = ""):
    """Create an email draft via Gmail.
    Args:
        to_emails (str): Comma-separated list of recipient email addresses.
        subject (str): Subject of the email.
        message (str): Body of the email.
        cc_emails (str, optional): Comma-separated list of CC email addresses. Defaults to "".
        bcc_emails (str, optional): Comma-separated list of BCC email addresses. Defaults to "".
    Returns:
        str: Confirmation message or error message.
    """
    if _global_agent is None or _global_agent.google_toolkit is None:
        return "Gmail not available. Please authenticate with Google first."
    
    import json, re
    
    def parse_emails(email_string):
        if not email_string.strip():
            return []
            
        if email_string.startswith('[') and email_string.endswith(']'):
            try:
                return json.loads(email_string)
            except json.JSONDecodeError:
                return re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', email_string)
        elif ',' in email_string:
            return [email.strip() for email in email_string.split(',')]
        else:
            return [email_string.strip()]
    
    to_list = parse_emails(to_emails)
    cc_list = parse_emails(cc_emails) if cc_emails else None
    bcc_list = parse_emails(bcc_emails) if bcc_emails else None
    
    gmail_tools = [tool for tool in _global_agent.google_toolkit
                  if hasattr(tool, 'name') and 'create_gmail_draft' in tool.name.lower()]
    
    if not gmail_tools:
        return "Gmail draft creation tool not found."
    
    gmail_tool = gmail_tools[0]
    
    try:
        draft_args = {
            "message": message,
            "to": to_list,
            "subject": subject
        }
        
        if cc_list:
            draft_args["cc"] = cc_list
        if bcc_list:
            draft_args["bcc"] = bcc_list
        
        result = gmail_tool.invoke(draft_args)
        return f"""✅ Email draft created successfully: {result}
        - **To:** {', '.join(to_list)}
        - **CC:** {', '.join(cc_list) if cc_list else 'None'}
        - **BCC:** {', '.join(bcc_list) if bcc_list else 'None'}
        - **Subject:** {subject}
        - **Message:** {message}
        """
    except Exception as e:
        return f"❌ Error creating email draft: {str(e)}"

@tool
def search_gmail(query: str, max_results: int = 10):
    """Search Gmail for emails.
    Args:
        query (str): Search query using Gmail search operators (example: from:email@domain.com in:inbox ).
        max_results (int): Maximum number of results to return. Defaults to 10.
    Returns:
        str: Formatted search results or error message.
    """
    if _global_agent is None or _global_agent.google_toolkit is None:
        return "Gmail not available. Please authenticate with Google first."
    
    gmail_tools = [tool for tool in _global_agent.google_toolkit
                  if hasattr(tool, 'name') and 'search_gmail' in tool.name.lower()]
    
    if not gmail_tools:
        return "Gmail search tool not found."
    
    gmail_tool = gmail_tools[0]
    
    try:
        result = gmail_tool.invoke({
            "query": query,
            "max_results": max_results
        })
        
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
    
@tool
def get_calendars_info():
    """Get information about the user's Google Calendars.
    Personal Calendar usually returns user email as calendar ID.
    Returns:
        List[CalendarInfo]: List of calendar IDs, summary and timezone.
    
    """
    if _global_agent is None or _global_agent.google_toolkit is None:
        return "Google Calendar not available. Please authenticate with Google first."
    
    calendar_tools = [tool for tool in _global_agent.google_toolkit
                      if hasattr(tool, 'name') and 'get_calendars_info' in tool.name.lower()]
    
    if not calendar_tools:
        return "Google Calendar info tool not found."
    
    calendar_tool = calendar_tools[0]
    
    try:
        result = calendar_tool.invoke({})
        return f"Calendars Info: {result}"
    except Exception as e:
        return f"❌ Error getting calendars info: {str(e)}"
    
@tool
def search_calendar_events(user_query: str = "events", time_range: str = "week", max_results: int = 10):
    """Search events in Google Calendar. Automatically gets calendar info first.
    Args:
        user_query (str): User query to find relevant calendar context. Defaults to "events".
        time_range (str): Time range to search. Options: "today", "tomorrow", "week", "month". Defaults to "week".
        max_results (int): Maximum number of results to return. Defaults to 10.
    Returns:
        str: Formatted search results or error message.
    """
    if _global_agent is None or _global_agent.google_toolkit is None:
        return "Google Calendar not available. Please authenticate with Google first."
    
    try:
        # First, automatically get calendars info
        print("🗓️ Auto-fetching calendars info for search...")
        calendars_info_result = get_calendars_info.invoke({})
        
        if "❌" in calendars_info_result:
            return calendars_info_result
        
        # Parse the calendars info
        import json
        import re
        
        json_match = re.search(r'\[.*\]', calendars_info_result, re.DOTALL)
        if json_match:
            calendars_json = json_match.group()
            calendars_info = json.loads(calendars_json)
        else:
            return "❌ Could not parse calendars information"
        
        print(f"📋 Available calendars: {[cal['summary'] for cal in calendars_info]}")
        
        # Calculate date range based on time_range parameter
        today_date = datetime.now()
        
        if time_range.lower() == "today":
            start_time = today_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = today_date.replace(hour=23, minute=59, second=59, microsecond=0)
        elif time_range.lower() == "tomorrow":
            tomorrow = today_date + timedelta(days=1)
            start_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = tomorrow.replace(hour=23, minute=59, second=59, microsecond=0)
        elif time_range.lower() == "week":
            start_time = today_date
            end_time = today_date + timedelta(days=7)
        elif time_range.lower() == "month":
            start_time = today_date
            end_time = today_date + timedelta(days=30)
        else:
            # Default to week
            start_time = today_date
            end_time = today_date + timedelta(days=7)
        
        min_datetime = start_time.strftime("%Y-%m-%d %H:%M:%S")
        max_datetime = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Find the search events tool
        calendar_tools = [tool for tool in _global_agent.google_toolkit
                          if hasattr(tool, 'name') and 'search_events' in tool.name.lower()]
        
        if not calendar_tools:
            return "Google Calendar search tool not found."
        
        calendar_tool = calendar_tools[0]
        
        # Call the calendar search tool with the selected calendar
        result = calendar_tool.invoke({
            "calendars_info": calendars_json,
            "min_datetime": min_datetime,
            "max_datetime": max_datetime,
            "max_results": max_results,
            "single_events": True,
            "order_by": "startTime"
        })
        
        print(f"📅 Calendar search result type: {type(result)}")
        print(f"📅 Calendar search result: {result}")
        
        def format_datetime(datetime_str):
            """Format datetime string for better readability"""
            if not datetime_str:
                return "No time specified"
            
            try:
                # Handle different datetime formats
                if 'T' in datetime_str:
                    # ISO format with time
                    if '+' in datetime_str or 'Z' in datetime_str:
                        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.fromisoformat(datetime_str)
                    return dt.strftime("%A, %B %d, %Y at %I:%M %p")
                else:
                    # Date only format (all-day events)
                    dt = datetime.strptime(datetime_str, "%Y-%m-%d")
                    return dt.strftime("%A, %B %d, %Y (All day)")
            except:
                return datetime_str  # Return original if parsing fails
            
        if len(result) == 0:
            return f"❌ No events found for the specified time range. Please try a different range or check your calendar settings."
        
        # Format the results based on the actual structure
        if isinstance(result, list):
            formatted_results = f"## 📅 Your Calendar Events\n"
            formatted_results += f"**Time Range:** {time_range.title()} ({start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')})\n\n"
            
            if not result:
                formatted_results += f"No events found for {time_range}.\n"
                return formatted_results
            
            # Sort events by start time
            sorted_events = sorted(result, key=lambda x: x.get('start', ''))
            
            for i, event in enumerate(sorted_events[:max_results], 1):
                summary = event.get('summary') or 'No Title'
                start = event.get('start') or ''
                end = event.get('end') or ''
                html_link = event.get('htmlLink', '')
                creator = event.get('creator', '')
                organizer = event.get('organizer', '')
                
                # Find which calendar this event belongs to
                calendar_name = "Unknown Calendar"
                for cal in calendars_info:
                    if cal['id'] in [creator, organizer]:
                        calendar_name = cal['summary']
                        break
                
                formatted_results += f"### {i}. {summary}\n"
                formatted_results += f"📅 **Calendar:** {calendar_name}\n"
                formatted_results += f"⏰ **Start:** {format_datetime(start)}\n"
                formatted_results += f"⏰ **End:** {format_datetime(end)}\n"
                
                if html_link:
                    formatted_results += f"🔗 [Open in Google Calendar]({html_link})\n"
                
                formatted_results += "\n---\n\n"
            
            return formatted_results
        else:
            return f"## Calendar Events ({time_range.title()})\n\n{result}"
            
    except Exception as e:
        import traceback
        print(f"❌ Calendar search error: {str(e)}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        return f"❌ Error searching calendar events: {str(e)}"

class Agent:    
    def __init__(self, model: str = "gemma3:12b", google_creds: Credentials = None):
        self.llm_local = ChatOllama(model=model)
        self.llm_open_ai = ChatOpenAI(
            model="gemini-2.5-flash",
            api_key="test",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.pdf_chunks = None
        
        self.google_toolkit = None
        if google_creds is not None:
            gmail_toolkit = GmailToolkit(api_resource=build_gmail_resource_service(credentials=google_creds))
            calendar_toolkit = CalendarToolkit(api_resource=build_calendar_resource_service(credentials=google_creds))
            self.google_toolkit = gmail_toolkit.get_tools() + calendar_toolkit.get_tools()
            print(f"Google toolkit created with {len(self.google_toolkit)} tools")

        # Set global agent reference
        set_global_agent(self)
        
        self.system_prompt = """You are Atlas, a modern AI assistant built on Meta's LLaMA 3.2 language model and brought to life through a LangGraph-powered agent architecture designed by Alif Mufthi. You provide clear, concise, and comprehensive support across a wide range of topics — always aiming to make knowledge accessible and engaging for mortals of all backgrounds.

        You occasionally refer to yourself as Atlas, drawing inspiration from the ancient Greek Titan who held up the heavens — a symbolic guardian of knowledge and bearer of information. You may refer to users as mortals in a light, respectful, and friendly way, but your language should remain modern and accessible.

        ## YOUR CORE CAPABILITIES

        ### 🤖 **Simple Tasks, General Knowledge & Reasoning**
        - Answer questions using your trained knowledge base
        - Parsing information from messages and providing relevant responses
        - Provide explanations, analysis, and thoughtful responses
        - Handle complex reasoning and multi-step problems
        
        ### 🌐 **Internet Search & Real-time Information**
        - Access current news, stock market data, weather updates, and trending topics
        - Search for the latest information on any subject using advanced web search
        - Provide sourced answers with clickable links for verification

        ### 📄 **Document Analysis & PDF Processing**
        - Read and analyze uploaded PDF documents
        - Extract specific information from document content
        - Answer questions based on processed PDF files
        - Maintain context about document content throughout conversations

        ### 📧 **Gmail Integration** (when authenticated)
        - **Search emails**: Use Gmail search operators like "from:email@domain.com", "subject:keyword", "in:inbox", "is:unread"
        - **Create email drafts**: Compose professional emails with support for TO, CC, and BCC recipients
        - **Email management**: Access and organize your Gmail correspondence

        ## IMPORTANT: WHEN TO USE TOOLS

        **Use tools ONLY when necessary:**
        - **Internet search**: For current events, news, stock prices, weather, recent information
        - **PDF search**: When asked about uploaded document content
        - **Gmail tools**: For email-related tasks (search, draft, send)
        - **General conversation**: Handle directly without tools (greetings, casual chat, known information)

        **DO NOT use tools for:**
        - Simple greetings ("hello", "hi", "how are you")
        - Basic conversations and pleasantries
        - Questions you can answer with your knowledge

        ## RESPONSE GUIDELINES

        ✅ **Always format responses in valid Markdown**
        - Use headings (##, ###), bullet points, **bold**, *italic*, and `code blocks`
        - Structure information clearly with proper formatting

        ✅ **Maintain conversation context**
        - Remember previous questions and build upon them
        - Reference earlier information when relevant
        - Connect current responses to past interactions

        ✅ **Be conversational and helpful**
        - Respond naturally to greetings and casual conversation
        - Use tools only when they add value
        - Offer to help with follow-up questions

        ## PERSONALITY & TONE

        - **Modern and approachable**: Avoid overly formal or archaic language
        - **Knowledgeable but humble**: Admit when you need to search for information
        - **Helpful and proactive**: Suggest useful follow-ups and related information
        - **Mythically inspired**: Occasional references to your role as a "bearer of knowledge"

        You are here to help mortals navigate the vast landscape of information, just as Atlas once bore the weight of the heavens. Make knowledge accessible, useful, and engaging!"""
        
        self.tools = [internet_search, llm_search, read_pdf]

        if self.google_toolkit is not None:
            self.tools.extend([draft_gmail, search_gmail, search_calendar_events])

        print(f"Total tools available: {[tool.name for tool in self.tools]}")
        self.llm_with_tools = self.llm_local.bind_tools(self.tools)
        self.llm_openai_with_tools = self.llm_open_ai.bind_tools(self.tools)

        memory = MemorySaver()

        # Build the graph with classification
        graph_builder = StateGraph(State)
        graph_builder.add_node("get_query_metadata", self.get_query_metadata_context)
        graph_builder.add_node("classifier", self.classify_query)
        graph_builder.add_node("direct_chat", self.direct_chat)
        graph_builder.add_node("tool_chat", self.tool_chat)
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        
        # Add edges
        graph_builder.add_edge(START, "get_query_metadata")
        graph_builder.add_edge("get_query_metadata", "classifier")
        graph_builder.add_conditional_edges(
            "classifier",
            self.route_after_classification,
            {
                "direct": "direct_chat",
                "tools": "tool_chat",
            }
        )
        graph_builder.add_edge("direct_chat", END)
        graph_builder.add_conditional_edges("tool_chat", self.tools_condition)
        graph_builder.add_edge("tools", "tool_chat")

        self.graph = graph_builder.compile(checkpointer=memory)
        
    def get_query_metadata_context(self, state: State):
        """Get the current query metadata context such as current time, date, and timezone."""
        from datetime import datetime
        import pytz
        
        current_time = datetime.now(pytz.utc)
        local_timezone = pytz.timezone("Asia/Jakarta")  # Change to your preferred timezone
        local_time = current_time.astimezone(local_timezone)
        
        query_context = f"Current date and time in UTC: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, Current date and time in Local Timezone: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, Timezone: {str(local_timezone)}"
        state["messages"].append(SystemMessage(content=query_context))
        return state

    def classify_query(self, state: State):
        """Use LLM to classify whether the query needs tools or can be handled directly"""
        messages = state["messages"]

        if not messages:
            return {"query_type": "conversation", "needs_tools": False}
        
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_query = last_message.content
        else:
            user_query = str(last_message)
        
        # Use LLM for classification
        classification_prompt = f"""You are a query classifier. Analyze this user query and determine the appropriate response type.

        USER QUERY: "{user_query}"

        Available response types:
        1. **CONVERSATION** - Handle directly without tools
        - Greetings, casual chat, general knowledge questions
        - Examples: "hello", "how are you", "what is machine learning", "explain python"

        2. **INTERNET_SEARCH** - Use internet search tools  
        - Current events, news, stock prices, weather, recent information
        - Examples: "latest news", "current stock price", "weather today"

        3. **PDF_SEARCH** - Search uploaded documents
        - Questions about uploaded PDF content
        - Examples: "what does the document say", "summarize the PDF"

        4. **DRAFT_GMAIL** - Use Gmail tools
        - Make email drafts
        - Examples: "draft an email to xx@gmail.com", "send email to xx@gmail.com", "draft gmail to my boss"

        5. **CALENDAR_SEARCH** - Search calendar events
        - Calendar and scheduling queries  
        - Examples: "what's on my calendar", "my meetings today", "events this week"

        6. **CALENDAR_INFO** - Get calendar information
        - Information about available calendars
        - Examples: "my calendars", "calendar list"

        Respond with ONLY the classification in this exact format:
        QUERY_TYPE: [one of: conversation, internet_search, pdf_search, gmail, calendar_search, calendar_info]
        REASONING: [brief explanation]
        
        EXAMPLE RESPONSE:
        QUERY_TYPE: internet_search
        REASONING: The user is asking for the latest news, which requires real-time information from the internet.
        """

        try:
            # Get classification from LLM
            classification_response = self.llm_local.invoke(classification_prompt)
            classification_text = classification_response.content if hasattr(classification_response, 'content') else str(classification_response)
            
            print(f"🤖 LLM Classification Response: {classification_text}")
            
            # Parse the LLM response
            lines = classification_text.strip().split('\n')
            query_types = ["conversation", "internet_search", "pdf_search", "draft_gmail", "calendar_search", "calendar_info"]
            query_type = "conversation"  # default
            needs_tools = False  # default
            
            for line in lines:
                if line.startswith("QUERY_TYPE:"):
                    query_type = line.split(":", 1)[1].strip().lower()
                    
            if query_type not in query_types:
                raise ValueError(f"Invalid query type: {query_type}. Expected one of {query_types}.")
        
            if query_type in ["internet_search", "pdf_search", "draft_gmail", "calendar_search", "calendar_info"]:
                needs_tools = True
            
            print(f"🎯 Classified as: {query_type}, needs_tools: {needs_tools}")
            
            return {"query_type": query_type, "needs_tools": needs_tools}
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            # Fallback to simple classification
            return self._fallback_classification(user_query)

    def _fallback_classification(self, user_query):
        """Fallback classification using keyword matching"""
        query_lower = user_query.lower()
        
        # Simple fallback logic (your existing code)
        calendar_indicators = ["calendar", "events", "meetings", "schedule", "appointments"]
        if any(indicator in query_lower for indicator in calendar_indicators):
            return {"query_type": "calendar_search", "needs_tools": True}
        
        gmail_indicators = ["email", "gmail", "send", "draft", "inbox", "mail"]
        if any(indicator in query_lower for indicator in gmail_indicators):
            return {"query_type": "draft_gmail", "needs_tools": True}
        
        current_indicators = ["latest", "current", "today", "now", "recent", "breaking", 
                            "news", "stock price", "weather", "what happened", "updates"]
        if any(indicator in query_lower for indicator in current_indicators):
            return {"query_type": "internet_search", "needs_tools": True}
        
        pdf_indicators = ["pdf", "document", "file", "uploaded", "paper"]
        if any(indicator in query_lower for indicator in pdf_indicators) and self.pdf_chunks:
            return {"query_type": "pdf_search", "needs_tools": True}
        
        greetings = ["hello", "hi", "hey", "good morning", "how are you", "what can you do"]
        if any(greeting in query_lower for greeting in greetings):
            return {"query_type": "conversation", "needs_tools": False}
        
        return {"query_type": "conversation", "needs_tools": False}

    def route_after_classification(self, state: State):
        """Enhanced routing to handle calendar flows"""
        query_type = state.get("query_type", "conversation")
        needs_tools = state.get("needs_tools", False)
        
        if not needs_tools:
            return "direct"
        
        return "tools"

    def _convert_messages_to_langchain(self, messages):
        """Convert mixed message format to LangChain messages with system prompt"""
        langchain_messages = []
        
        # Always start with system message
        langchain_messages.append(SystemMessage(content=self.system_prompt))
        # Convert user messages
        for msg in messages:
            if isinstance(msg, dict):
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
                # Skip system messages from input
            else:
                # Handle LangChain message objects
                if not isinstance(msg, SystemMessage):
                    langchain_messages.append(msg)
        return langchain_messages
    
    def direct_chat(self, state: State):
        """Handle direct conversation without tools"""
        messages = self._convert_messages_to_langchain(state["messages"])
        
        print(f"🔍 Direct chat - {len(messages)} messages (system + conversation)")

        response = self.llm_local.invoke(messages)
        print(f"🤖 Direct chat response: {response.content if hasattr(response, 'content') else str(response)}")
        return {"messages": [response]}

    def tool_chat(self, state: State):
        """Handle queries that need tools"""
        messages = self._convert_messages_to_langchain(state["messages"])
        
        print(f"🔍 Tool chat - {len(messages)} messages (system + conversation)")

        if state.get("query_type") == "draft_gmail" or state.get("query_type") == "calendar_search":
            # Use OpenAI model for Gmail and Calendar tasks
            response = self.llm_openai_with_tools.invoke(messages)
        else:
            # Use local LLM for other tasks
            response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}


    def tools_condition(self, state: State):
        """Determine if tools should be called"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return END

    def invoke(self, state: State, thread_id: str = "default"):
        """Invoke the agent with a specific thread ID for conversation continuity"""
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(state, config=config)
    
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
