import random
from typing import List
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from chatbot import Chatbot, ChunkEvent, Message, Role, SourcesEvent, create_history
from pdf_loader import load_uploaded_file, cleanup_got_model
import time
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
    page_title='RAG',
    page_icon='😊',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Private-RAG")
st.subheader("Private intelligence for your thoughts and files")

def get_file_cache_key(files: List[UploadedFile]) -> str:
    """Generate cache key from file names and sizes"""
    file_info = [(f.name, f.size) for f in files]
    file_info.sort()  # sort for consistent ordering
    return str(file_info)

@st.cache_resource(show_spinner=False)
def create_chatbot_cached(cache_key: str, files: List[UploadedFile]):
    """Create chatbot with proper caching"""
    files = [load_uploaded_file(f) for f in files]
    return Chatbot(files)

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
    

uploaded_files = show_uploaded_documents()
if uploaded_files:
    cache_key = get_file_cache_key(uploaded_files)
    chatbot = create_chatbot_cached(cache_key, uploaded_files)
    cleanup_got_model()

if "messages" not in st.session_state:
    st.session_state.messages = create_history(WELCOME_MESSAGE)

with st.sidebar:
    st.title("Your files")
    files_list_text = "\n".join([f"- {file.name}" for file in chatbot.files])
    st.markdown(files_list_text)

for message in st.session_state.messages:
    avatar = "🧑‍💻" if message.role == Role.USER else "🤖"
    with st.chat_message(message.role.value, avatar=avatar):
        st.markdown(message.content)

if prompt := st.chat_input("Type your message"):
    with st.chat_message('user', avatar='🧑‍💻'):
        st.markdown(prompt)

    with st.chat_message('assistant', avatar='🤖'):
        # create layout
        status_placeholder = st.empty()
        
        metrics_cols = st.columns(3)
        retrieval_time_metric = metrics_cols[0].empty()
        docs_retrieved_metric = metrics_cols[1].empty()
        docs_relevant_metric = metrics_cols[2].empty()
        
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
        
        for event in chatbot.ask(prompt, st.session_state.messages):
            if isinstance(event, SourcesEvent):
                # update metrics
                retrieval_time = time.time() - start_time
                retrieval_time_metric.metric("Retrieval", f"{retrieval_time:.2f}s")
                docs_retrieved_metric.metric("Retrieved", len(event.content))
                num_sources = len(event.content)
                
                status_placeholder.empty()
                
                if not sources_shown and event.content:
                    sources_shown = True
                    sources_header.markdown("### Retrieved Sources")
                    
                    with sources_container:
                        # create tabs for different sources
                        if len(event.content) <= 3:
                            # if few sources, use expanders
                            for i, doc in enumerate(event.content):
                                source_name = doc.metadata.get('source', 'Unknown')
                                
                                with st.expander(f"📄 {source_name} (Excerpt {i+1})", expanded=(i==0)):
                                    # show metadata
                                    st.caption(f"*Source: {source_name}*")
                                    
                                    # show content
                                    content = doc.page_content
                                    if len(content) > 500:
                                        st.markdown(content[:500] + "...")
                                        if st.button(f"Show full content", key=f"full_{i}"):
                                            st.markdown(content)
                                    else:
                                        st.markdown(content)
                        else:
                            # if many sources, use tabs
                            tabs = st.tabs([f"Source {i+1}" for i in range(len(event.content))])
                            
                            for i, (tab, doc) in enumerate(zip(tabs, event.content)):
                                with tab:
                                    source_name = doc.metadata.get('source', 'Unknown')
                                    st.caption(f"*{source_name}*")
                                    st.markdown(doc.page_content)
            
            if isinstance(event, ChunkEvent):
                if not sources_shown:
                    status_placeholder.empty()
                
                # show answer header once generating starts
                if not full_response:
                    answer_header.markdown("### 💬 Answer")
                
                chunk = event.content
                full_response += chunk
                
                current_time = time.time()
                if not hasattr(st.session_state, 'last_update_time'):
                    st.session_state.last_update_time = current_time
                
                time_since_update = current_time - st.session_state.last_update_time
                
                if time_since_update >= 0.5:  
                    message_placeholder.markdown(full_response + "▌")
                    st.session_state.last_update_time = current_time
        
        message_placeholder.markdown(full_response)
        
        # update final metrics
        total_time = time.time() - start_time
        if sources_shown:
            docs_relevant_metric.metric("Relevant", num_sources)
        
        # add feedback buttons at the end
        st.markdown("---")
        feedback_cols = st.columns([1, 1, 8])
        with feedback_cols[0]:
            if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                st.success("Thanks for your feedback!")
        with feedback_cols[1]:
            if st.button("👎 Not helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                st.info("Thanks! We'll improve.")
                