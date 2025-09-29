# filepath: autogen-literature-review/autogen-literature-review/src/main.py
import streamlit as st
import os
import arxiv
from typing import List, Dict
from dotenv import load_dotenv
import json
import time
import google.generativeai as genai

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

SELECTED_MODEL = None

def get_available_gemini_models() -> List[str]:
    """Return a list of model names available to the API key."""
    try:
        models = list(genai.list_models())
        names = []
        for m in models:
            name = getattr(m, "name", "")
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            if name and ("generateContent" in methods or not methods):
                names.append(name)
        return sorted(names)
    except Exception:
        return []

def setup_gemini() -> bool:
    """Configure Gemini client using free API key from env. Returns True if OK."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found. Add it to your .env or Streamlit secrets.")
        return False
    try:
        genai.configure(api_key=api_key)
        # Pick an available model that supports generateContent
        candidates = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.0-pro",
            "gemini-pro",
        ]
        available = set(get_available_gemini_models())

        global SELECTED_MODEL
        # Prefer list_models if it worked
        for name in candidates:
            if not available or name in available:
                try:
                    # Probe the model once with a trivial call
                    _ = genai.GenerativeModel(name)
                    SELECTED_MODEL = name
                    break
                except Exception:
                    continue
        if not SELECTED_MODEL:
            st.error("No compatible Gemini model found for your key. Verify quota and model availability in AI Studio.")
            return False
        st.info(f"Using Gemini model: {SELECTED_MODEL}")
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        return False

def summarize_with_gemini(paper: Dict) -> str:
    """Use Gemini to summarize an arXiv paper in a strict template."""
    prompt = f"""
You are a research assistant specialized in summarizing academic papers.
Follow this exact structure for the provided paper details. Do NOT return JSON.

---
### Title: {paper.get('title','')}

**Author Names:** {', '.join(paper.get('authors', []))}  
**Publication Details:** {paper.get('published','')} (arXiv)  

**Abstract:** {paper.get('summary','')}

Using the abstract and any relevant background knowledge, produce the rest:

**Description:** <summary in your own words>  

**Scope:** <scope of research>  

**Methodology:** <technical approaches, models, algorithms>  

**Research Gaps:** <limitations / gaps>  

**Research Questions:** <questions addressed>  

**Important Points:**  
- Point 1  
- Point 2  

**Important Sentences (direct quotes):**  
1. "..."  
2. "..."  

**Results & Conclusion:** <findings, statistics, contributions>  

**Advantages:** <strengths>  
**Disadvantages:** <limitations>  
---
"""
    model = genai.GenerativeModel(SELECTED_MODEL or "gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text or ""

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
        gem_key = os.getenv("GEMINI_API_KEY")
        if gem_key:
            st.success("✅ Gemini API Key loaded")
        else:
            st.error("❌ Gemini API Key missing")
            st.info("Add GEMINI_API_KEY to your .env file")

        # Model selection (optional manual override)
        available_models: List[str] = []
        if gem_key:
            try:
                genai.configure(api_key=gem_key)
                available_models = get_available_gemini_models()
            except Exception:
                available_models = []
        default_models = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-1.0-pro",
        ]
        model_options = available_models or default_models
        selected = st.selectbox("Gemini model (auto if not changed):", model_options, index=0)
        if selected:
            # Remember user choice
            st.session_state["user_selected_model"] = selected
        
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
        
        # Configure Gemini and set model
        user_selected = st.session_state.get("user_selected_model")
        if user_selected:
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                global SELECTED_MODEL
                SELECTED_MODEL = user_selected
            except Exception as e:
                st.error(f"Failed to select model {user_selected}: {e}")
                return
        elif not setup_gemini():
            return

        st.session_state.is_running = True
        st.session_state.results = []

        status_placeholder = st.empty()
        results_placeholder = st.empty()

        try:
            with st.spinner("Searching arXiv…"):
                papers = arxiv_search(topic, max_results=max(10, num_papers * 3))
                papers = papers[:num_papers]

            summaries = []
            for idx, paper in enumerate(papers, start=1):
                status_placeholder.markdown(
                    f'<p class="status-running">🔄 Summarizing paper {idx}/{len(papers)}…</p>',
                    unsafe_allow_html=True
                )
                try:
                    text = summarize_with_gemini(paper)
                except Exception as e:
                    text = f"Summary failed: {e}"
                summaries.append({"paper": paper, "summary": text})

                with results_placeholder.container():
                    st.markdown("### 📨 Latest Result")
                    with st.expander(paper["title"], expanded=True):
                        st.write(text)

            st.session_state.results = summaries
            status_placeholder.markdown(
                '<p class="status-complete">✅ Literature Review Complete!</p>',
                unsafe_allow_html=True
            )
            st.session_state.is_running = False
            st.success(f"🎉 Successfully completed literature review on '{topic}' with {len(summaries)} papers!")

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
            
            for i, item in enumerate(st.session_state.results):
                title = item["paper"]["title"] if isinstance(item, dict) else f"Result {i+1}"
                with st.expander(title, expanded=i == len(st.session_state.results)-1):
                    st.write(item.get("summary", ""))
        
        with tab2:
            st.markdown("### 💬 Detailed Agent Communication")
            
            for i, item in enumerate(st.session_state.results):
                st.markdown(f"""
                <div class="result-container">
                    <h4>Paper {i+1}</h4>
                    <pre>{item.get('summary','')}</pre>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 📁 Export Results")
            
            # Prepare data for export
            export_data = {
                "topic": topic if 'topic' in locals() else "Unknown",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_papers_requested": num_papers if 'num_papers' in locals() else 5,
                "messages": [item.get('summary','') for item in st.session_state.results]
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
