from typing import Any, Iterator, List, Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from llama_cpp import Llama
from pydantic import PrivateAttr
import time


class ChatLlamaCppWrapper(BaseChatModel):
    """
    LangChain-compatible wrapper for llama-cpp-python's Llama class.
    """

    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_batch: int = 512
    n_threads: Optional[int] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    streaming: bool = True
    verbose: bool = False
    _llm: Optional[Llama] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialize the Llama model
        print(f"Loading llama.cpp model: {self.model_path}")
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            verbose=self.verbose,
            flash_attn=True,
            # auto-detect chat format from GGUF metadata 
        )
        print(f"Loaded model with {self.n_ctx} context window")

    def _convert_messages_to_dict(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to llama.cpp chat format."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            else:
                # fallback for other message types
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response (non-streaming)."""
        # convert messages to llama.cpp format
        formatted_messages = self._convert_messages_to_dict(messages)
        # generate response
        response = self._llm.create_chat_completion(
            messages=formatted_messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            stream=False,
            stop=stop,
        )
        # extract content
        content = response['choices'][0]['message']['content']
        # create generation
        generation = ChatGeneration(message=AIMessage(content=content))

        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the response (for streaming support)."""
        # convert messages to llama.cpp format
        formatted_messages = self._convert_messages_to_dict(messages)
        # stream response
        stream = self._llm.create_chat_completion(
            messages=formatted_messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            stream=True,
            stop=stop,
        )

        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta and delta['content']:
                content = delta['content']
                # yield chunk
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=content)
                )
                # callback
                if run_manager:
                    run_manager.on_llm_new_token(content)

    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "llama-cpp-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
