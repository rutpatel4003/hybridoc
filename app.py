import random
from typing import List
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from chatbot import Chatbot, ChunkEvent, Message, Role, SourcesEvent, create_history
from pdf_loader import load_uploaded_file
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

st.header("RAG")
st.subheader("Private intelligence for your thoughts and files")

@st.cache_resource(show_spinner=False)
def create_chatbot(files: List[UploadedFile]):
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
chatbot = create_chatbot(uploaded_files)

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
        full_response = ''
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state='running')
        for event in chatbot.ask(prompt, st.session_state.messages):
            if isinstance(event, SourcesEvent):
                for i, doc in enumerate(event.content):
                    with st.expander(f"Source #{i+1}"):
                        st.markdown(doc.page_content)
            if isinstance(event, ChunkEvent):
                chunk = event.content
                full_response += chunk
                message_placeholder.markdown(full_response)
                