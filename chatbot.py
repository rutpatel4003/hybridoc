from pdf_loader import File
from data_ingestor import ingest_files
from typing import List, TypedDict, Iterable, Literal
from enum import Enum
from config import Config
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from pdf_loader import File
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import json

close_tag = '</think>'
tag_length = len(close_tag)

@dataclass 
class SourcesEvent: 
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

class State(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer: str
    retry_count: int 


SYSTEM_PROMPT = """
You are a highly precise technical expert. Answer the question using ONLY the provided context.
- START your answer immediately with the facts.
- DO NOT use filler phrases like "Based on the context" or "According to the excerpts."
- If the answer is not in the context, state: "Information not found in document."
- Format math using LaTeX.
- Do NOT use outside knowledge.
""".strip()

PROMPT = """
Here's the information you have about the excerpts of the files:

<context>
{context}
</context>

One file can have multiple excerpts.

Please respond to the query below

<question>
{question}
</question>

Answer:
"""

FILE_TEMPLATE = """
<file>
    <name>{name}</name>
    <content>{content}</content>
</file>
""".strip()

GRADER_SYSTEM_PROMPT = """
You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
The question does not have to be exactly the same or too specific when checking it with the context for it to be relevant, it can be a broad question with similar semantic meaning too if it makes sense. For example, a question might be based for mathematical reasoning, so check if the context contains the mathematical terms, do not just discard it if it looks gibberish without proper reasoning.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            SYSTEM_PROMPT
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', PROMPT)
    ]
)

class Role(Enum):
    USER = 'user'
    ASSISTANT = 'assistant'

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class ChunkEvent:
    content: str

@dataclass
class SourcesEvent:
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

def _remove_thinking_from_message(message: str) -> str:
    # handle cases where the tag might not exist
    if close_tag in message:
        # find the end of the tag and then .lstrip() to remove 
        return message[message.find(close_tag) + tag_length:].lstrip()
    return message.strip()

def create_history(welcome_message: Message) -> List[Message]:
    return [welcome_message]

class Chatbot:
    def __init__(self, files: List[File]):
        self.files = files
        self.retriever = ingest_files(files)
        self.llm = ChatOllama(model=Config.Model.NAME,
                            temperature=Config.Model.TEMPERATURE,
                            num_ctx=4096,
                            num_predict=1028, 
                            num_thread=8,      
                            verbose=False,
                            keep_alive=-1,
                            streaming=True,
                            callbacks=[StreamingStdOutCallbackHandler()])
        self.workflow = self._create_workflow()

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(FILE_TEMPLATE.format(name=doc.metadata['source'], content=doc.page_content) for doc in docs)
    
    def _retrieve(self, state: State):
        print(f"RETRIEVING: {state['question']}")
        context = self.retriever.invoke(state['question'])
        return {"context": context}
    
    def _grade_documents(self, state: State):
        """
        Filter out irrelevant documents.
        """
        print('Document Relevance Checking in Process!')
        question = state['question']
        documents = state['context']
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:500]}" for i, doc in enumerate(documents) 
        ]) # batching documents to query faster
        prompt = ChatPromptTemplate.from_messages([
            ('system', """You are a document grader. For each document, decide if it's relevant.
            Return a JSON Array with the format defined below:
            [
                {{"doc_id": 1, "relevant": true}},
                {{"doc_id": 2, "relevant": false}}
            ]
            Use double braces in the response.
            """),
            ('human', 'Question: {question}\n\nDocuments:\n{docs_text}'),
        ])
        grader_chain = prompt | self.llm | StrOutputParser()
        try:
            result = grader_chain.invoke({'question': question, 'docs_text': docs_text})
            scores = json.loads(result)
            score_map = {item['doc_id']: item.get('relevant', False) for item in scores}    
            filtered_docs = [
                doc for i, doc in enumerate(documents) 
                if score_map.get(i + 1, False) # default to False (remove) if LLM didn't mention it
            ]
            
            print(f"Filtered: {len(filtered_docs)}/{len(documents)} relevant")
            return {'context': filtered_docs}
        except Exception as e:
            print(f'Grading failed: {e}')
            return {'context': documents}
    
    def _transform_query(self, state: State):
        """
        Transform the query to produce a better question
        """
        print("Transforming Query!")
        question = state['question']
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are generating a better search query for a vector database. Rephrase the input question to be more specific. Just rephrase the input question, do not add any preamble or explanation, your output should only contain the rephrased question, nothing else."),
            ("human", f"Look at the input and try to reason about the underlying semantic intent / meaning. \n\n Initial Question: {question} \n\n Formulate an improved question: "),
        ])
        chain = prompt | self.llm | StrOutputParser()
        better_question = chain.invoke({})
        current_retry = state.get('retry_count', 0)
        return {'question': better_question, 'retry_count': current_retry+1}

    def _generate(self, state: State):
        print('Generating Answer!')
        messages = PROMPT_TEMPLATE.invoke(
            {
                "question": state['question'],
                "context": self._format_docs(state['context']),
                'chat_history': state['chat_history'],
            }
        )
        answer = self.llm.invoke(messages)
        return {"answer": answer}
    
    def _decide_to_generate(self, state: State) -> Literal['_transform_query', '_generate']:
        """
        Determines whether to generate an answer or re-generate a question.
        """
        filtered_documents = state['context']
        retry_count = state.get('retry_count', 0)
        # if no relevant documents and not self-corrected a lot of times
        if not filtered_documents and retry_count<1: # loop 1 time
            print('Decision: Documents irrelevant, rerouting to transform!')
            return '_transform_query'
        else:
            print('Decision: Generating Answer!')
            return '_generate'

    
    def _condense_question(self, state: State):
        """
        Takes the chat history and the current question, rewrites the question to be standalone so the vector store can understand it. 
        """
        chat_history = state.get('chat_history', [])
        question = state['question']
        
        # if no chat history exists, return the question as it is 
        if not chat_history:
            return {"question": question}
        condense_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", condense_system_prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        reformulated_question = chain.invoke({
            "chat_history": chat_history,
            "question": question
        })
        
        # update the state with the new question
        return {"question": reformulated_question}
    
    def _create_workflow(self) -> CompiledStateGraph:
        graph_builder = StateGraph(State)
        if not Config.Chatbot.GRADING_MODE:
            graph_builder.add_sequence([self._condense_question, self._retrieve, self._generate])
            graph_builder.add_edge(START, '_condense_question')
            graph_builder.add_edge('_generate', END)
        else:
            graph_builder.add_sequence([self._condense_question, self._retrieve, self._grade_documents])
            graph_builder.add_node('_transform_query', self._transform_query)
            graph_builder.add_node('_generate', self._generate)
            graph_builder.add_edge(START, '_condense_question')
            graph_builder.add_conditional_edges(
                '_grade_documents', self._decide_to_generate,
                {
                    '_transform_query': '_transform_query',
                    '_generate': '_generate'
                },
            )
            graph_builder.add_edge('_transform_query', '_retrieve')
            graph_builder.add_edge('_generate', END)
        return graph_builder.compile()

    def _ask_model(
            self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        history = [
            AIMessage(m.content) if m.role == Role.ASSISTANT else HumanMessage(m.content) for m in chat_history
        ]
        payload = {"question": prompt, "chat_history": history, 'retry_count': 0}

        config = {
            "configurable": {"thread_id": 42}
        }

        for event_type, event_data in self.workflow.stream(
            payload, config=config, stream_mode=['updates', 'messages']
        ):
            if event_type =='messages':
                chunk, metadata = event_data
                if metadata.get('langgraph_node') == '_generate':
                    if chunk.content:
                        yield ChunkEvent(chunk.content)
            if event_type == 'updates':
                if "_retrieve" in event_data:
                    documents = event_data['_retrieve']['context']
                    unique_docs = []
                    seen_content = set()
                    for doc in documents:
                        if doc.page_content not in seen_content:
                            unique_docs.append(doc)
                            seen_content.add(doc.page_content)
                    yield SourcesEvent(unique_docs)
                if "_generate" in event_data:
                    answer = event_data['_generate']['answer']
                    yield FinalAnswerEvent(answer.content)

    def ask(self, prompt: str, chat_history: List[Message]) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        for event in self._ask_model(prompt, chat_history):
            yield event
            if isinstance(event, FinalAnswerEvent):
                response = _remove_thinking_from_message("".join(event.content))
                chat_history.append(Message(role=Role.USER, content=prompt))
                chat_history.append(Message(role=Role.ASSISTANT, content=response))
