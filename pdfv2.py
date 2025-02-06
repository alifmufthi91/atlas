import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader  
from langchain_experimental.text_splitter import SemanticChunker  
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def main():
    # Streamlit file uploader  
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")  

    # If a file is uploaded
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load PDF
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()

        # Split text into semantic chunks
        text_splitter = SemanticChunker(HuggingFaceEmbeddings())  
        documents = text_splitter.split_documents(docs)

        # Generate embeddings 
        embeddings = HuggingFaceEmbeddings()  
        vector_store = FAISS.from_documents(documents, embeddings) 

        # Connect Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Connect LLM
        llm = OllamaLLM(model="deepseek-r1")

        # Craft the prompt template  
        prompt = """  
        1. Use ONLY the context below.  
        2. If unsure, say "I don’t know".  
        3. Keep answers under 4 sentences.  

        Context: {context}  

        Question: {input}  

        Answer:  
        """  
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

        # Chain 1: Generate answers  
        # llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT) 
        # llm_chain = QA_CHAIN_PROMPT | llm 

        # Chain 2: Combine document chunks  
        # document_prompt = PromptTemplate(  
        #     template="Context:\ncontent:{page_content}\nsource:{source}",  
        #     input_variables=["page_content", "source"]  
        # )  
        
        document_prompt = PromptTemplate.from_template(
            "Context:\ncontent:{page_content}\nsource:{source}"
        )

        # llm_chain = create_stuff_documents_chain(llm_chain, document_prompt)

        # system_prompt = (
        #     "You are an assistant for question-answering tasks. "
        #     "Use the following pieces of retrieved context to answer "
        #     "the question."
        #     "\n\n"
        #     "{context}"
        # )

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system_prompt),
        #         ("human", "{input}"),
        #     ]
        # )

        question_answer_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT, document_prompt=document_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Streamlit UI  
        user_input = st.text_input("Ask your PDF a question:")  

        if user_input:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_input})
                st.write(response['answer'])

if __name__ == "__main__":
    main()