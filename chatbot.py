import streamlit as st
from openai import OpenAI
import re

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
def process_ollama_response(ollama_output):
    """
    Parses Ollama output, separates thinking sections and response,
    and formats thinking sections as code markdown.
    """
    # Regex to capture content within <think> tags and the actual response
    match = re.match(r"(<think>(.*?)</think>)?(.*)", ollama_output, re.DOTALL)
    
    think_content = match.group(2) if match.group(1) else None
    response_content = match.group(3).strip()  # Remove leading/trailing whitespace

    return think_content, response_content


def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.title("🤖 Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message(msg["role"]):
                if "thinking" in msg:
                    st.code(msg["thinking"], language="markdown")
                st.write(msg["content"])

    st.empty()
    if prompt := st.chat_input("Enter a prompt"):
        # if not openai_api_key:
        #     st.info("Please add your OpenAI API key to continue.")
        #     st.stop()

        # connect to local ollama
        client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
        # with st.container(border=True):
            st.write(prompt)

        with st.chat_message("assistant"):
        # with st.container(border=True):
            with st.spinner("Thinking...", show_time=True):
                response = client.chat.completions.create(model="deepseek-r1", messages=st.session_state.messages)
                msg = response.choices[0].message.content
                thinking, answer = process_ollama_response(msg)
            st.code(thinking, language="markdown")
            st.write(answer)
            st.empty()

        # put the code block thinking and answer to session state
        st.session_state.messages.append({"role": "assistant", "content": answer, "thinking": thinking})


if __name__ == "__main__":
    main()