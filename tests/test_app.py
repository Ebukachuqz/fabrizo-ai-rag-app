from dataclasses import dataclass, field
from pathlib import Path

import lancedb
import pandas as pd
import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from lancedb.embeddings import TextEmbeddingFunction, get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker
from streamlit.testing.v1 import AppTest

import init_lancedb
from src import config, prompt, rag as rag_module, session, vectordb
from src.langgraph_workflow import initialize_rag_workflow

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # Never use developer credentials or contact paid services in regression tests.
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "DG_API_KEY", "GROQ_MODEL",
                 "OPENAI_MODEL", "LANCEDB_PATH", "LANCEDB_TABLE"):
        monkeypatch.delenv(name, raising=False)
    st.cache_resource.clear()


@dataclass
class RecordingModel:
    model_name: str = "test-model"
    calls: list = field(default_factory=list)
    error: Exception = None

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=f"{self.model_name}: Chelsea signed a striker. [source](https://example.com/tweet)")


def test_shared_database_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    expected = (str(config.PROJECT_ROOT / "lancedb_data"), "fabrizio_tweets")
    assert init_lancedb.get_database_settings() == expected
    assert vectordb.get_database_settings() == expected
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "custom-db"))
    monkeypatch.setenv("LANCEDB_TABLE", "custom_tweets")
    assert init_lancedb.get_database_settings() == vectordb.get_database_settings()
    assert vectordb.get_database_settings() == (str(tmp_path / "custom-db"), "custom_tweets")


@pytest.mark.parametrize("choice", ["Groq", "OpenAI"])
def test_real_provider_constructor_has_no_forwarded_stream(choice):
    model = prompt.initialize_llm(choice, "test-key")
    assert model.streaming is False
    assert "stream" not in model.model_kwargs
    if choice == "Groq":
        assert model.model_name == "qwen/qwen3.8-27b"


def test_openai_never_falls_back_to_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        prompt.initialize_llm("OpenAI")
    assert prompt.get_api_key("Groq", " sidebar-key ") == "sidebar-key"


@pytest.mark.parametrize("stage", ["retrieval", "model"])
def test_rag_propagates_original_exception(monkeypatch, stage):
    error = RuntimeError(f"Original {stage} failure")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(rag_module, "lancedb_hybrid_search", fail if stage == "retrieval" else lambda q: (["tweet"], ["url"]))
    model = RecordingModel(error=error)
    with pytest.raises(RuntimeError) as caught:
        rag_module.rag("1", initialize_rag_workflow(model), "Chelsea?")
    assert caught.value is error


def test_model_and_key_switch_preserve_conversation(monkeypatch):
    created = []
    def create(choice, key):
        model = RecordingModel(model_name=choice)
        created.append(model)
        return model
    monkeypatch.setattr(session, "initialize_llm", create)
    state = {}
    first = session.get_session_workflow(state, "Groq", "first-key")
    prompt.ask_llm("1", first, "First question", "First context")
    assert session.get_session_workflow(state, "Groq", "first-key") is first
    assert len(created) == 1
    second = session.get_session_workflow(state, "OpenAI", "second-key")
    assert second is not first
    prompt.ask_llm("1", second, "Follow-up question", "Second context")
    assert "First question" in [message.content for message in created[-1].calls[-1]]
    third = session.get_session_workflow(state, "OpenAI", "replacement-key")
    assert third is not second
    other_state = {}
    other = session.get_session_workflow(other_state, "Groq", "first-key")
    assert other_state["rag_memory"] is not state["rag_memory"]
    assert other is not first


def test_long_conversation_summarizes_without_losing_latest_answer():
    model = RecordingModel()
    workflow = initialize_rag_workflow(model)
    for number in range(5):
        answer, chunks = prompt.ask_llm("1", workflow, f"Question {number}", "Tweet context")
        assert answer == "".join(chunks)
    saved = workflow.get_state({"configurable": {"thread_id": "1"}}).values
    assert saved["summary"]
    assert len(saved["messages"]) <= 6
    assert any("Previous conversation summary:" in call[0].content for call in model.calls)


@get_registry().register("test-football")
class FootballEmbeddings(TextEmbeddingFunction):
    def ndims(self):
        return 3

    def generate_embeddings(self, texts):
        return [[float("chelsea" in text.lower()), float("arsenal" in text.lower()), 1.0] for text in texts]


@pytest.fixture
def database(monkeypatch, tmp_path):
    embedding = FootballEmbeddings.create()
    class Tweet(LanceModel):
        tweet_count: int
        tweet_id: int
        username: str
        text: str = embedding.SourceField()
        created_at: str
        url: str
        vector: Vector(3) = embedding.VectorField()

    (tmp_path / "data").mkdir()
    pd.DataFrame([
        {"tweet_count": 1, "tweet_id": 1, "username": "FabrizioRomano", "text": "Chelsea signed a striker", "created at": "2024-09-01", "url": "https://example.com/tweet"},
        {"tweet_count": 2, "tweet_id": 2, "username": "FabrizioRomano", "text": "Arsenal signed a defender", "created at": "2024-09-02", "url": "https://example.com/arsenal"},
    ]).to_csv(tmp_path / "data" / "fabrizioromano_tweets.csv", index=False)
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setenv("LANCEDB_TABLE", "test_tweets")
    monkeypatch.setattr(init_lancedb, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(init_lancedb, "get_tweet_schema", lambda: Tweet)
    monkeypatch.setattr(vectordb, "get_reranker", lambda: RRFReranker())
    return init_lancedb.initialize_database()


def test_real_database_ingestion_hybrid_search_and_rag(database):
    assert database.count_rows() == 2
    assert init_lancedb.initialize_database().count_rows() == 2
    context, urls = vectordb.lancedb_hybrid_search("Chelsea", k=1)
    assert "Chelsea signed a striker" in context[0]
    assert urls == ["https://example.com/tweet"]
    model = RecordingModel()
    answer, chunks, sources = rag_module.rag("1", initialize_rag_workflow(model), "Chelsea")
    assert answer == "".join(chunks)
    assert "https://example.com/tweet" in sources
    assert "Chelsea signed a striker" in model.calls[0][0].content


def test_ui_opens_without_credentials_or_database(monkeypatch):
    def forbidden():
        pytest.fail("Startup must not build the database")
    monkeypatch.setattr(init_lancedb, "initialize_database", forbidden)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Fabrizio Romano Q&A AI Chatbot"
    app.chat_input[0].set_value("Hi there").run()
    assert not app.exception
    assert "GROQ_API_KEY" in app.error[0].value


def test_ui_chat_reruns_and_model_switch(database, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setattr(session, "initialize_llm", lambda choice, key: RecordingModel(choice))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.chat_input[0].set_value("Chelsea").run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["chat_history"][-1].content.startswith("Groq:")
    app.sidebar.selectbox[0].select("OpenAI").run()
    app.chat_input[0].set_value("And Arsenal?").run(timeout=30)
    assert not app.exception and not app.error
    assert app.session_state["chat_history"][-1].content.startswith("OpenAI:")
    assert len(app.session_state["chat_history"]) == 5


def test_ui_shows_original_retrieval_error(database, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setattr(session, "initialize_llm", lambda choice, key: RecordingModel())
    def fail(query):
        raise RuntimeError("Search index is unavailable")
    monkeypatch.setattr(rag_module, "lancedb_hybrid_search", fail)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.chat_input[0].set_value("Chelsea").run(timeout=30)
    assert not app.exception
    assert "Search index is unavailable" in app.error[0].value
    assert "unpack" not in app.error[0].value


def test_ui_voice_failure_keeps_text_answer(database, monkeypatch):
    from src import text2speech
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("DG_API_KEY", "test-deepgram")
    monkeypatch.setattr(session, "initialize_llm", lambda choice, key: RecordingModel())
    def fail(*args, **kwargs):
        raise RuntimeError("Speech service unavailable")
    monkeypatch.setattr(text2speech, "text2speech", fail)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.checkbox[0].check().run()
    app.chat_input[0].set_value("Chelsea").run(timeout=30)
    assert not app.exception and not app.error
    assert "audio playback is unavailable" in app.warning[0].value
    assert isinstance(app.session_state["chat_history"][-1], AIMessage)
