# filepath: autogen-literature-review/autogen-literature-review/src/main.py
import streamlit as st
from autogen_agentchat.agents import AssistantAgent
import asyncio
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import arxiv
from typing import List, Dict
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AutoGen Literature Review Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .result-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .paper-item {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4285f4;
    }
    
    .status-running {
        color: #ff6b35;
        font-weight: bold;
    }
    
    .status-complete {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

def arxiv_search(query: str, max_results: int = 10) -> List[Dict]:
    """Return a compact list of arXiv papers matching query."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: List[Dict] = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
        )
    return papers

def initialize_agents():
    """Initialize the AutoGen agents."""
    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment variables!")
        st.info("Please add your Gemini API key to your .env file or Streamlit secrets.")
        return None, None, None
    
    GEMINI_brain = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=api_key,
    )

    arxiv_agent = AssistantAgent(
        name="arxiv_agent",
        description="An agent that helps with searching and retrieving academic papers from arXiv.",
        model_client=GEMINI_brain,
        tools=[arxiv_search],
        system_message=(
            '''Given a user topic, think of the best arXiv query and call the
            provided tool. Always fetch five times the papers requested so
            that you can down-select the most relevant ones. When the tool
            returns, choose exactly the number of papers requested and pass
            them as concise JSON to the summarizer'''
        )
    )

    litreview_agent = AssistantAgent(
    name="litreviewer",
    model_client=GEMINI_brain,
    description="Agent that helps with literature review tasks.",
    system_message="""
You are a research assistant specialized in summarizing academic papers.
For each paper, output the following **exact structured format**:

---
### Title: <paper title>

**Author Names:** <list of authors>  
**Publication Details:** <year, venue>  

**Abstract:** <exact abstract from paper>  

**Description:** <summary in your own words>  

**Scope:** <scope of research>  

**Methodology:** <technical approaches, models, algorithms>  

**Research Gaps:** <limitations / gaps>  

**Research Questions:** <questions addressed>  

**Important Points:**  
- Point 1  
- Point 2  
...  

**Important Sentences (direct quotes):**  
1. "..."  
2. "..."  

**Results & Conclusion:** <findings, statistics, contributions>  

**Advantages:** <strengths>  
**Disadvantages:** <limitations>  
---

⚠️ Always output in this exact structure. Do NOT return JSON. Do NOT include tool logs.
"""
)

    team = RoundRobinGroupChat(
        participants=[arxiv_agent, litreview_agent],
        max_turns=2
    )
    
    return arxiv_agent, litreview_agent, team

async def run_literature_review(team, topic: str, num_papers: int = 5):
    """Run the literature review with the given topic and number of papers."""
    task = f'Conduct a literature review on the topic {topic} and return exactly {num_papers} papers'
    
    messages = []
    async for message in team.run_stream(task=task):
        messages.append(message)
        yield message

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 AutoGen Literature Review Assistant</h1>
        <p>Powered by Multi-Agent AI System for Academic Research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key status
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.success("✅ Gemini API Key loaded")
        else:
            st.error("❌ Gemini API Key missing")
            st.info("Add GEMINI_API_KEY to your .env file")
        
        st.header("📋 Agent Information")
        
        # Agent cards
        st.markdown("""
        <div class="agent-card">
            <h4>🔍 ArXiv Agent</h4>
            <p>Searches and retrieves academic papers from arXiv based on your topic.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <h4>📝 Literature Review Agent</h4>
            <p>Analyzes papers and creates comprehensive literature reviews with detailed summaries.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clear results button
        if st.button("🗑️ Clear Results", type="secondary"):
            st.session_state.results = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Research Topic")
        topic = st.text_input(
            "Enter your research topic:",
            placeholder="e.g., machine learning, natural language processing, computer vision",
            help="Be specific about your research area for better results"
        )
    
    with col2:
        st.header("📊 Parameters")
        num_papers = st.selectbox(
            "Number of papers:",
            options=[3, 5, 8, 10],
            index=1,
            help="More papers = more comprehensive review but longer processing time"
        )
    
    # Start button
    if st.button("🚀 Start Literature Review", type="primary", disabled=st.session_state.is_running):
        if not topic:
            st.error("Please enter a research topic!")
            return
        
        if not api_key:
            st.error("Please configure your Gemini API key!")
            return
        
        # Initialize agents
        with st.spinner("Initializing agents..."):
            arxiv_agent, litreview_agent, team = initialize_agents()
        
        if team is None:
            return
        
        st.session_state.is_running = True
        st.session_state.results = []
        
        # Create placeholder for real-time updates
        status_placeholder = st.empty()
        results_placeholder = st.empty()
        
        # Run the literature review
        try:
            async def run_review():
                messages = []
                async for message in run_literature_review(team, topic, num_papers):
                    messages.append(message)
                    
                    # Update status
                    status_placeholder.markdown(
                        f'<p class="status-running">🔄 Processing... ({len(messages)} messages received)</p>',
                        unsafe_allow_html=True
                    )
                    
                    # Display latest message in results
                    with results_placeholder.container():
                        st.markdown("### 📨 Latest Agent Activity")
                        if message:
                            with st.expander(f"Message from {getattr(message, 'source', 'Agent')}", expanded=True):
                                st.write(str(message))
                
                return messages
            
            # Run async function
            messages = asyncio.run(run_review())
            
            # Update final status
            status_placeholder.markdown(
                '<p class="status-complete">✅ Literature Review Complete!</p>',
                unsafe_allow_html=True
            )
            
            # Store results
            st.session_state.results = messages
            st.session_state.is_running = False
            
            st.success(f"🎉 Successfully completed literature review on '{topic}' with {num_papers} papers!")
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.session_state.is_running = False
    
    # Display results
    if st.session_state.results and not st.session_state.is_running:
        st.markdown("---")
        st.header("📄 Literature Review Results")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Summary", "💬 Agent Messages", "📁 Export"])
        
        with tab1:
            st.markdown("### 🔍 Review Summary")
            
            # Try to extract and display structured information
            for i, message in enumerate(st.session_state.results):
                with st.expander(f"Result {i+1}", expanded=i == len(st.session_state.results)-1):
                    st.write(str(message))
        
        with tab2:
            st.markdown("### 💬 Detailed Agent Communication")
            
            for i, message in enumerate(st.session_state.results):
                st.markdown(f"""
                <div class="result-container">
                    <h4>Message {i+1}</h4>
                    <pre>{str(message)}</pre>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 📁 Export Results")
            
            # Prepare data for export
            export_data = {
                "topic": topic if 'topic' in locals() else "Unknown",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_papers_requested": num_papers if 'num_papers' in locals() else 5,
                "messages": [str(msg) for msg in st.session_state.results]
            }
            
            # JSON download
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📄 Download as JSON",
                data=json_str,
                file_name=f"literature_review_{export_data['topic'].replace(' ', '_')}_{export_data['timestamp'].replace(':', '-').replace(' ', '_')}.json",
                mime="application/json"
            )
            
            # Text download
            text_content = f"""Literature Review Results
Topic: {export_data['topic']}
Generated: {export_data['timestamp']}
Papers Requested: {export_data['num_papers_requested']}

{'='*50}
AGENT MESSAGES
{'='*50}

"""
            for i, msg in enumerate(export_data['messages']):
                text_content += f"\nMessage {i+1}:\n{'-'*20}\n{msg}\n\n"
            
            st.download_button(
                label="📝 Download as Text",
                data=text_content,
                file_name=f"literature_review_{export_data['topic'].replace(' ', '_')}_{export_data['timestamp'].replace(':', '-').replace(' ', '_')}.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using AutoGen Multi-Agent Framework and Streamlit</p>
        <p>Powered by Gemini AI • arXiv API • Academic Research</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()