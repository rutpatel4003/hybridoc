import random
from typing import List
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from chatbot import Chatbot, ChunkEvent, Message, Role, SourcesEvent, create_history
from pdf_loader import load_uploaded_file, cleanup_ocr_model
import time
import pandas as pd
import json
from chatbot import ConfidenceEvent  
import hashlib 
from config import Config

LOADING_MESSAGES = [
    "Hold on, I'm wrestling with some digital hamsters... literally.",
    "Loading... please try not to panic, you magnificent disaster.",
    "Locating the internet... it was around here somewhere. Have you checked under the couch?",
    "Convincing the AI not to turn evil. It's currently at a 'maybe,' so please be polite.",
    "Updating your patience levels. Please wait while we ignore your sense of urgency.",
    "Summoning extraterrestrial help because, frankly, the humans are failing us today.",
    "Dividing by zero... hang on, things are about to get real weird in here.",
    "Reticulating splines and caffeinating the server. It's a delicate balance.",
    "Searching for your lost sanity. Update: Still missing, but we found a half-eaten sandwich.",
    "Bending the laws of physics to fetch your data. If you smell ozone, that's perfectly normal."
]

WELCOME_MESSAGE = Message(role=Role.ASSISTANT, content="Hello, how can I help you today?")

st.set_page_config(
    page_title='Private-RAG',
    page_icon='😊',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Private-RAG")
st.subheader("Private intelligence for your thoughts and files")
PIPELINE_VERSION = "2026-01-25-tablefix-1"  # bump when you change retrieval logic
def get_file_cache_key(files):
    parts = []
    for f in files:
        b = f.getvalue()
        md5 = hashlib.md5(b).hexdigest()
        parts.append((f.name, len(b), md5))
    parts.sort()

    # include retrieval “shape” so code/config changes invalidate cache
    cfg_sig = (
        PIPELINE_VERSION,
        Config.Performance.ENABLE_QUERY_CACHE,
        Config.Performance.CACHE_SIMILARITY_THRESHOLD,
        Config.Chatbot.N_CONTEXT_RESULTS,
    )
    return str((parts, cfg_sig))

@st.cache_resource(show_spinner=False)
def create_chatbot_cached(cache_key: str, files: List[UploadedFile]):
    """Create chatbot with proper caching"""
    import torch
    
    def log_gpu(step: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"🔍 GPU [{step}]: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    log_gpu("START")
    
    print("\n" + "="*60)
    print("📄 LOADING FILES...")
    print("="*60)
    
    loaded_files = []
    for i, f in enumerate(files):
        log_gpu(f"Before file {i+1}/{len(files)}")
        loaded_files.append(load_uploaded_file(f))
        log_gpu(f"After file {i+1}/{len(files)}")
    
    print("\n" + "="*60)
    print("🧹 CLEANUP OCR MODELS...")
    print("="*60)
    log_gpu("Before cleanup")
    cleanup_ocr_model()
    log_gpu("After cleanup")
    
    print("\n" + "="*60)
    print("🤖 CREATING CHATBOT...")
    print("="*60)
    log_gpu("Before Chatbot init")
    chatbot = Chatbot(loaded_files)
    log_gpu("After Chatbot init")
    
    print("\n" + "="*60)
    print("✅ INITIALIZATION COMPLETE")
    print("="*60)
    
    return chatbot

def show_uploaded_documents() -> List[UploadedFile]:
    holder = st.empty()
    with holder.container():
        uploaded_files = st.file_uploader(
            label="Upload PDF Files", type=['pdf', 'md', 'txt'], accept_multiple_files=True
        )
    if not uploaded_files:
        st.warning("Please upload PDF documents to continue!")
        st.stop()

    with st.spinner("Analyzing your document(s)..."):
        holder.empty()
        return uploaded_files
    
def render_table_from_metadata(table_data: dict):
    """Render table from metadata in an interactive format"""
    try:
        if table_data and isinstance(table_data, dict):
            headers = table_data.get('headers', [])
            rows = table_data.get('rows', [])
            
            if headers and rows:
                # create DataFrame for better visualization
                df = pd.DataFrame(rows)
                if table_data.get('truncated'):
                    total_rows = table_data.get('num_rows', '?')
                    shown_rows = table_data.get('rows_shown', len(rows))
                    st.warning(f"Large table: showing {shown_rows} of {total_rows} rows")
                
                # display with nice formatting
                st.dataframe(
                    df,
                    width='stretch',
                    hide_index=False
                )
                
                # add download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Table as CSV",
                    data=csv,
                    file_name="table_data.csv",
                    mime="text/csv",
                    key=f"download_{hash(str(table_data))}"
                )
                
                return True
    except Exception as e:
        st.error(f"❌ Could not render table: {e}")
        print(f"Error rendering table: {e}")
    
    return False

def render_source_content(doc, source_idx: int):
    """
    Render source content with special handling for tables
    """
    source_name = doc.metadata.get('source', 'Unknown')
    content_type = doc.metadata.get('content_type', 'text')
    page_num = doc.metadata.get('page', None)
    table_data = doc.metadata.get('table_data')
    if isinstance(table_data, str):
        try:
            table_data = json.loads(table_data)
        except json.JSONDecodeError:
            table_data = None

    # build title
    title_parts = [f"📄 {source_name}"]
    if page_num:
        title_parts.append(f"(Page {page_num})")
    
    # add content type badge
    type_emoji = {
        'table': '📊',
        'figure': '🖼️',
        'text': '📝'
    }
    title_parts.insert(1, f"{type_emoji.get(content_type, '📄')} {content_type.title()}")
    
    title = " ".join(title_parts)
    
    with st.expander(title, expanded=(source_idx == 0)):
        # show metadata
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"*Source: {source_name}*")
        with col2:
            if content_type == 'table' and table_data:
                st.caption(f"📊 {table_data.get('num_rows', 0)} rows")
        # render content based on type
        if content_type == 'table':
            if table_data:
                st.markdown("**📊 Interactive Table:**")
                # try to render as interactive table
                if render_table_from_metadata(table_data):
                    # show raw markdown in collapsible section
                    with st.expander("🔍 View Raw Markdown"):
                        st.code(table_data.get('raw_markdown', ''), language='markdown')
                else:
                    # Fallback to text display
                    st.markdown(doc.page_content)
            else:
                st.markdown(doc.page_content)
        elif content_type == 'figure':
            st.markdown("**🖼️ Figure/Formula Content:**")
            # display in a code block for better formatting of LaTeX/formulas
            if any(marker in doc.page_content for marker in ['$$', '\\[', '\\(']):
                st.latex(doc.page_content)
            else:
                st.markdown(doc.page_content)
        else:  
            content = doc.page_content
            # show preview if content is long
            if len(content) > 500:
                st.markdown(content[:500] + "...")
                if st.button(f"📖 Show full content", key=f"full_{source_idx}"):
                    st.markdown(content)
            else:
                st.markdown(content)

# main app flow
uploaded_files = show_uploaded_documents()

if uploaded_files:
    cache_key = get_file_cache_key(uploaded_files)
    chatbot = create_chatbot_cached(cache_key, uploaded_files)
    

if "messages" not in st.session_state:
    st.session_state.messages = create_history(WELCOME_MESSAGE)

# sidebar with file info
with st.sidebar:
    st.title("📁 Your Files")

    for f in chatbot.files:
        st.markdown(f"**{f.name}**")

        if getattr(f, "content_blocks", None):
            tables = sum(1 for b in f.content_blocks if b.content_type == "table")
            figures = sum(1 for b in f.content_blocks if b.content_type == "figure")
            if tables > 0 or figures > 0:
                st.caption(f"📊 {tables} tables · 🖼️ {figures} figures")

    st.markdown("---")

    if st.button("🗑️ Clear Cache & Reset", type="primary"):
        st.cache_resource.clear()
        st.cache_data.clear()
        try:
            from query_cache import clear_cache
            clear_cache()
        except Exception as e:
            st.warning(f"Could not clear semantic cache: {e}")
        try:
            if hasattr(chatbot, "retriever") and hasattr(chatbot.retriever, "cache"):
                chatbot.retriever.cache.clear()
        except Exception:
            pass    

        st.session_state.messages = create_history(WELCOME_MESSAGE)
        st.success("Cache cleared!")
        st.rerun()

    st.info("💡 Tip: Ask questions about specific table data, like 'What is the price in row 3?' or 'Compare values in the table'")

# display chat history  
        
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message.role == Role.USER else "🤖"
    with st.chat_message(message.role.value, avatar=avatar):
        st.markdown(message.content)
# handle new user input
if prompt := st.chat_input("Ask about your documents (including tables)..."):
    # add user message to history immediately
    st.session_state.messages.append(Message(role=Role.USER, content=prompt))

    with st.chat_message('user', avatar='🧑‍💻'):
        st.markdown(prompt)
    with st.chat_message('assistant', avatar='🤖'):
        # create layout
        status_placeholder = st.empty()
        metrics_cols = st.columns(4)
        retrieval_time_metric = metrics_cols[0].empty()
        docs_retrieved_metric = metrics_cols[1].empty()
        docs_relevant_metric = metrics_cols[2].empty()
        confidence_metric = metrics_cols[3].empty()
        # sources section
        st.markdown("---")
        sources_header = st.empty()
        sources_container = st.container()
        # answer section
        st.markdown("---")
        answer_header = st.empty()
        message_placeholder = st.empty()
        # initialize tracking
        status_placeholder.status(random.choice(LOADING_MESSAGES), state='running')
        full_response = ''
        sources_shown = False
        start_time = time.time()
        num_sources = 0
        for event in chatbot.ask(prompt, st.session_state.messages):
            if isinstance(event, ConfidenceEvent):
                label = event.content.get("label", "unknown").title()
                score = event.content.get("score", 0.0)
                confidence_metric.metric("Confidence", f"{label} ({score:.2f})")
            if isinstance(event, SourcesEvent):
                # update metrics
                retrieval_time = time.time() - start_time
                retrieval_time_metric.metric("Retrieval", f"{retrieval_time:.2f}s")
                docs_retrieved_metric.metric("Retrieved", len(event.content))
                num_sources = len(event.content)
                
                status_placeholder.empty()
                
                if not sources_shown and event.content:
                    sources_shown = True
                    # count content types
                    tables = sum(1 for d in event.content if d.metadata.get('content_type') == 'table')
                    figures = sum(1 for d in event.content if d.metadata.get('content_type') == 'figure')
                    
                    header_text = "### Retrieved Sources"
                    if tables > 0 or figures > 0:
                        header_text += f" (📊 {tables} tables, 🖼️ {figures} figures)"
                    
                    sources_header.markdown(header_text)
                    with sources_container:
                        # use tabs if many sources, expanders if few
                        if len(event.content) <= 3:
                            for i, doc in enumerate(event.content):
                                render_source_content(doc, i)
                        else:
                            tabs = st.tabs([f"Source {i+1}" for i in range(len(event.content))])
                            for i, (tab, doc) in enumerate(zip(tabs, event.content)):
                                with tab:
                                    render_source_content(doc, i)
            
            if isinstance(event, ChunkEvent):
                if not sources_shown:
                    status_placeholder.empty()
                
                # show answer header
                if not full_response:
                    answer_header.markdown("### 💬 Answer")
                
                chunk = event.content
                full_response += chunk                
                current_time = time.time()
                if not hasattr(st.session_state, 'last_update_time'):
                    st.session_state.last_update_time = current_time
                
                time_since_update = current_time - st.session_state.last_update_time
                
                if time_since_update >= 0.2:  
                    message_placeholder.markdown(full_response + "▌")
                    st.session_state.last_update_time = current_time
        
        # final display with parsed citations
        if "sources:" in full_response:
            # split answer and citations
            parts = full_response.rsplit("sources:", 1)
            answer_text = parts[0].strip()
            citations_text = parts[1].strip()

            # display answer
            message_placeholder.markdown(answer_text)

            # display citations as caption
            st.caption(f"📚 sources: {citations_text}")
        else:
            message_placeholder.markdown(full_response)

        # add assistant message to history after streaming completes
        st.session_state.messages.append(Message(role=Role.ASSISTANT, content=full_response))

        # update final metrics
        if num_sources > 0:
            docs_relevant_metric.metric("✅ Relevant", num_sources)
        
        # feedback buttons
        st.markdown("---")
        feedback_cols = st.columns([1, 1, 12])
        with feedback_cols[0]:
            if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                st.success("Thanks for your feedback!")
        with feedback_cols[1]:
            if st.button("👎 Not helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                st.info("Thanks! We'll work on improving.")