import streamlit as st
from main import run_assistant
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

st.set_page_config(page_title="IntelliFin Analyst", layout="centered")

# Title and description
st.title("💼 IntelliFin Analyst (Groq-Powered)")
st.write("An intelligent financial analysis assistant using your integrated financial systems.")

# Initialize LLM for post-processing
llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")

def process_output(llm, raw_output, query):
    prompt = PromptTemplate(
        input_variables=["query", "output"],
        template="""
        You are an intelligent financial assistant. Based on the query and the raw output received, provide a clean and concise answer for the user.
        
        Query: {query}
        Raw Output: {output}
        
        Instructions:
        - Remove any <think> sections or any internal processing explanations.
        - Summarize the relevant information from the raw output.
        - Provide only the relevant and structured answer to the user query.
        - Ensure clarity, professionalism, and coherence in your response.
        """
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.run({"query": query, "output": raw_output})
    return result

# Input box for user query
query = st.text_input("🧑‍💼 Ask a question:")

if st.button("Submit"):
    if query:
        with st.spinner('Processing...'):
            raw_response = run_assistant(query)
            final_response = process_output(llm, raw_response, query)
        
        st.write("### 💡 Answer:")
        st.write(final_response)
    else:
        st.error("Please enter a query to proceed.")
