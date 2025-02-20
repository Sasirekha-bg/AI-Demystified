import os
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import streamlit as st

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
class BlogState(TypedDict):
    topic: str
    title: str
    content: str

def generate_title(state: BlogState) -> dict:
    llm = ChatGroq(model="qwen-2.5-32b")  
    prompt = f"Generate a simple, clear, and engaging title for a blog about {state['topic']} (Machine Learning, Deep Learning, Agentic AI, or Generative AI). The title should be beginner-friendly and convey the core idea of the topic in a way that sparks curiosity and is easy to understand for a wide audience."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"title": response.content}

def generate_content(state: BlogState) -> dict:
    llm = ChatGroq(model="qwen-2.5-32b")
    prompt = f"Write a beginner-friendly blog in the style of Hay Alammar, explaining the topic of {state['topic']} (Machine Learning, Deep Learning, Agentic AI, or Generative AI). The blog should focus on making the topic simple and understandable to someone with no prior knowledge of AI. Use analogies, real-world examples, and avoid technical jargon. The tone should be friendly, engaging, and encouraging, with clear step-by-step explanations. The goal is to make the reader feel comfortable and excited to learn more"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"content": response.content}



# Create the StateGraph
workflow = StateGraph(BlogState)

# Add nodes to the graph
workflow.add_node("generate_title", generate_title)
workflow.add_node("generate_content", generate_content)

# Define edges (flow of execution)
workflow.add_edge(START, "generate_title")  # Start with the title generation
workflow.add_edge("generate_title", "generate_content")  # Then generate content
workflow.add_edge("generate_content", END)  # End the workflow

# Compile the graph
app = workflow.compile()

def generate_blog(topic: str) -> dict:
    # Define the initial state
    initial_state = {"topic": topic, "title": "", "content": ""}
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    return result

import streamlit as st


# Streamlit app
st.title("AI Demystified: Blogs That Make Learning Fun")
st.write("Enter a topic, and I'll make it simple for you!")

# Input: Topic
topic = st.text_input("Enter a topic:")

# Button: Generate Blog
if st.button("Teach Me"):
    if topic:
        with st.spinner("Generating blog..."):
            # Call your blog generation function
            blog = generate_blog(topic)
            
            # Display the results
            st.subheader("Title")
            st.write(blog["title"])
            
            st.subheader("Content")
            st.write(blog["content"])
    else:
        st.error("Please enter a topic!")
