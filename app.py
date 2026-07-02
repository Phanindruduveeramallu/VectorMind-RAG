import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

# Load environment variables
load_dotenv()

# Initialize the Gemini client
client = genai.Client()

# Pydantic Blueprint for structured data
class ResearchData(BaseModel):
    topic: str = Field(description="The main topic of research.")
    key_facts: List[str] = Field(description="A list of 3 distinct, high-quality facts about the topic.")
    source_type: str = Field(description="The general source classification, e.g., 'Technical Docs'.")

# --- AGENT 1: The Researcher ---
def researcher_agent(user_topic: str) -> ResearchData:
    prompt = f"Perform high-level research on the following topic and extract key facts: {user_topic}"
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResearchData,
            temperature=0.2
        ),
    )
    return ResearchData.model_validate_json(response.text)

# --- AGENT 2: The Writer ---
def writer_agent(data: ResearchData) -> str:
    prompt = f"""
    You are an expert technical documentation writer. Take the following structured research data and compile it into a beautifully formatted Markdown report.
    Use clear headings, bold text, and professional bullet points. Do not mention that you are an AI.
    
    Research Data:
    - Topic: {data.topic}
    - Source Category: {data.source_type}
    - Key Facts: {", ".join(data.key_facts)}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7)
    )
    return response.text

# --- STREAMLIT WEB UI ---
st.set_page_config(page_title="AI Multi-Agent Researcher", page_icon="🤖", layout="centered")

st.title("🤖 GenAI Multi-Agent Research Platform")
st.caption("Built with Python, Google Gemini 2.5, and Pydantic")
st.write("Enter any topic below. Two specialized AI agents will collaborate sequentially to research and write a professional report.")

# User Input Text Box
user_topic = st.text_input("Enter Research Topic:", placeholder="e.g., Quantum Computing, Blockchain, Docker...")

# Run Button
if st.button("Generate Report", type="primary"):
    if user_topic.strip() == "":
        st.warning("Please enter a valid topic first!")
    else:
        # Step 1: Agent 1 Logic with loading spinner
        with st.spinner("🕵️ Agent 1 (Researcher) is analyzing and validating facts..."):
            try:
                structured_facts = researcher_agent(user_topic)
                
                # Show the intermediate step in an expandable box
                with st.expander("✅ View Agent 1's Structured JSON"):
                    st.json(structured_facts.model_dump())
                    
            except Exception as e:
                st.error(f"Agent 1 failed: {e}")
                st.stop()

        # Step 2: Agent 2 Logic with loading spinner
        with st.spinner("✍️ Agent 2 (Writer) is compiling the markdown report..."):
            try:
                final_report = writer_agent(structured_facts)
            except Exception as e:
                st.error(f"Agent 2 failed: {e}")
                st.stop()

        # Success message and final output rendering
        st.success("✨ Report Generation Complete!")
        st.markdown("---")
        st.markdown(final_report)