"""Dependency-free checks of the actual functions with controlled substitutes.

Run: python tests/pseudo_smoke.py
These checks do not launch Streamlit, embed tweets, or call external services.
"""

import ast
from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import sha256
import logging
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    return namespace


def load_retrieval():
    namespace = {"Counter": Counter, "defaultdict": defaultdict, "csv": csv,
                 "lru_cache": lru_cache, "math": math, "os": os, "re": re,
                 "unicodedata": unicodedata, "PROJECT_ROOT": ROOT}
    tree = ast.parse((ROOT / "src/retrieval.py").read_text(encoding="utf-8"))
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "STOP_WORDS"
                         for target in node.targets)]
    exec(compile(ast.Module(body=constants, type_ignores=[]), "src/retrieval.py", "exec"), namespace)
    return load_functions("src/retrieval.py", namespace)


class HumanMessage:
    def __init__(self, content):
        self.content = content


class AIMessage(HumanMessage):
    pass


class PseudoSmokeTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.constructed_models = []

        def model(**kwargs):
            self.constructed_models.append(kwargs)
            return SimpleNamespace(model_name=kwargs["model"])

        self.prompt = load_functions("src/prompt.py", {
            "os": os, "ChatGroq": model, "ChatOpenAI": model,
            "HumanMessage": HumanMessage, "AIMessage": AIMessage,
        })

    def test_python_syntax(self):
        files = [ROOT / "app.py", ROOT / "init_lancedb.py"]
        for directory in ("src", "utils", "tests"):
            files.extend((ROOT / directory).glob("*.py"))
        for path in files:
            with self.subTest(file=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_database_defaults_overrides_and_blank_values(self):
        config = load_functions("src/config.py", {"os": os, "PROJECT_ROOT": ROOT})
        settings = config["get_database_settings"]
        expected = (str(ROOT / "lancedb_data"), "fabrizio_tweets")
        self.assertEqual(settings(), expected)
        os.environ.update(LANCEDB_PATH="custom-db", LANCEDB_TABLE="custom-table")
        self.assertEqual(settings(), ("custom-db", "custom-table"))
        os.environ.update(LANCEDB_PATH="", LANCEDB_TABLE="")
        self.assertEqual(settings(), expected)
        for file in ("init_lancedb.py", "src/vectordb.py"):
            tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
            self.assertTrue(any(
                isinstance(node, ast.ImportFrom) and node.module == "src.config"
                and any(alias.name == "get_database_settings" for alias in node.names)
                for node in tree.body
            ))

    def test_successful_rag_returns_three_values_and_context(self):
        calls = []
        def ask(*args):
            calls.append(args)
            return "Answer", ["An", "swer"]
        namespace = load_functions("src/rag.py", {
            "search_tweets": lambda query: (["tweet one", "tweet two"], ["url"]),
            "ask_llm": ask,
        })
        self.assertEqual(namespace["rag"]("thread", "workflow", "query"),
                         ("Answer", ["An", "swer"], ["url"]))
        self.assertEqual(calls[0], ("thread", "workflow", "query", "tweet one\ntweet two"))

    def test_original_retrieval_and_model_errors_are_preserved(self):
        for stage in ("retrieval", "model"):
            with self.subTest(stage=stage):
                original = RuntimeError(f"Original {stage} error")
                def fail(*args):
                    raise original
                namespace = load_functions("src/rag.py", {
                    "search_tweets": fail if stage == "retrieval" else lambda query: (["tweet"], ["url"]),
                    "ask_llm": fail if stage == "model" else lambda *args: ("answer", ["answer"]),
                })
                with self.assertRaises(RuntimeError) as raised:
                    namespace["rag"]("1", None, "question")
                self.assertIs(raised.exception, original)

    def test_current_groq_model_and_correct_streaming_parameter(self):
        os.environ["GROQ_API_KEY"] = "dummy-groq-key"
        self.prompt["initialize_llm"]("Groq")
        parameters = self.constructed_models[-1]
        self.assertEqual(parameters["model"], "qwen/qwen3.8-27b")
        self.assertIs(parameters["streaming"], False)
        self.assertNotIn("stream", parameters)
        os.environ["GROQ_MODEL"] = "configured-model"
        self.assertEqual(self.prompt["get_model_name"]("Groq"), "configured-model")

    def test_provider_credentials_do_not_fall_back_to_another_provider(self):
        os.environ["GROQ_API_KEY"] = "dummy-groq-key"
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            self.prompt["initialize_llm"]("OpenAI")
        self.prompt["initialize_llm"]("OpenAI", " dummy-openai-key ")
        self.assertEqual(self.constructed_models[-1]["api_key"], "dummy-openai-key")
        self.assertNotIn("stream", self.constructed_models[-1])
        self.assertEqual(self.prompt["get_api_key"]("Groq", " sidebar-key "), "sidebar-key")

    def test_ask_llm_uses_existing_workflow_and_returns_latest_answer(self):
        calls = []
        def invoke(state, config):
            calls.append((state, config))
            return {"messages": [AIMessage("Old answer"), HumanMessage("query"), AIMessage("New answer")]}
        answer, chunks = self.prompt["ask_llm"]("session-1", SimpleNamespace(invoke=invoke), "query", "context")
        self.assertEqual(answer, "New answer")
        self.assertEqual(answer, "".join(chunks))
        self.assertEqual(calls[0][1], {"configurable": {"thread_id": "session-1"}})
        self.assertEqual(calls[0][0]["context"], "context")
        self.assertEqual(self.constructed_models, [])

    def test_provider_key_and_model_switch_rebuild_workflow_and_keep_memory(self):
        workflows = []
        def build(llm, memory):
            workflow = {"llm": llm, "memory": memory}
            workflows.append(workflow)
            return workflow
        namespace = load_functions("src/session.py", {
            "sha256": sha256, "MemorySaver": object, "initialize_rag_workflow": build,
            "get_api_key": self.prompt["get_api_key"],
            "get_model_name": self.prompt["get_model_name"],
            "initialize_llm": self.prompt["initialize_llm"],
        })
        get_workflow = namespace["get_session_workflow"]
        state = {}
        first = get_workflow(state, "Groq", "first-key")
        self.assertIs(get_workflow(state, "Groq", "first-key"), first)
        self.assertEqual(len(self.constructed_models), 1)
        second = get_workflow(state, "OpenAI", "second-key")
        third = get_workflow(state, "OpenAI", "replacement-key")
        os.environ["OPENAI_MODEL"] = "configured-model"
        fourth = get_workflow(state, "OpenAI", "replacement-key")
        self.assertEqual(len(workflows), 4)
        for workflow in (second, third, fourth):
            self.assertIs(workflow["memory"], first["memory"])
        another = get_workflow({}, "Groq", "first-key")
        self.assertIsNot(another["memory"], first["memory"])
        self.assertEqual(state["rag_thread_id"], "1")

    def test_watcher_workaround_and_optional_audio(self):
        configuration = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
        self.assertEqual(configuration["server"]["fileWatcherType"], "none")
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(call.func, ast.Name) and call.func.id == "TemporaryDirectory" for call in calls))
        voice_branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                            and isinstance(node.test, ast.Name) and node.test.id == "read_aloud")
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "text2speech" for node in ast.walk(voice_branch)))

    def test_streamlit_pin_includes_session_message_fix(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("streamlit==1.45.1", requirements)

    def test_greetings_do_not_swallow_football_questions(self):
        greeting = load_retrieval()["get_greeting_response"]
        for query in ("hi", "Hi there!", "HELLO", "hey", "good morning"):
            with self.subTest(query=query):
                self.assertTrue(greeting(query).startswith("Hi!"))
        self.assertIsNone(greeting("Hi, what about Chelsea transfers?"))

    def test_keyword_search_ranks_and_cites_without_model_downloads(self):
        os.environ["RETRIEVAL_MODE"] = "keyword"
        namespace = load_retrieval()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            path = root / "data/fabrizioromano_tweets.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["text", "created at", "url"])
                writer.writeheader()
                writer.writerows([
                    {"text": "Chelsea signed a striker", "created at": "2024-09-01", "url": "https://example.com/chelsea"},
                    {"text": "Arsenal signed a defender", "created at": "2024-09-02", "url": "https://example.com/arsenal"},
                    {"text": "Rúben Amorim joined Manchester United", "created at": "2024-09-03", "url": "https://example.com/amorim"},
                ])
            namespace["PROJECT_ROOT"] = root
            self.assertEqual(namespace["get_retrieval_mode"](), "keyword")
            context, urls = namespace["search_tweets"]("What did Chelsea do?", k=1)
            self.assertEqual(urls, ["https://example.com/chelsea"])
            self.assertIn("Chelsea signed a striker", context[0])
            self.assertIn("2024-09-01", context[0])
            namespace["search_tweets"]("Chelsea", k=1)
            self.assertEqual(namespace["load_keyword_index"].cache_info().misses, 1)
            self.assertGreaterEqual(namespace["load_keyword_index"].cache_info().hits, 1)
            self.assertEqual(namespace["keyword_search"]("Ruben Amorim", k=1)[1], ["https://example.com/amorim"])
            self.assertEqual(namespace["keyword_search"]("nonexistentplayer"), ([], []))
            os.environ["RETRIEVAL_MODE"] = "invalid"
            with self.assertRaisesRegex(ValueError, "RETRIEVAL_MODE"):
                namespace["get_retrieval_mode"]()

    def test_no_matching_tweets_does_not_invoke_model(self):
        def forbidden(*args):
            self.fail("No-context responses must not invoke the model")
        namespace = load_functions("src/rag.py", {
            "search_tweets": lambda query: ([], []), "ask_llm": forbidden,
        })
        answer, chunks, urls = namespace["rag"]("1", None, "unmatched")
        self.assertIn("couldn't find matching tweets", answer)
        self.assertEqual(answer, "".join(chunks))
        self.assertEqual(urls, [])

    def test_actual_app_greeting_bypasses_database_and_provider(self):
        class State(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        class FakeStreamlit:
            def __init__(self):
                self.session_state = State()
                self.sidebar = self
                self.output = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def image(self, *args, **kwargs): pass
            def title(self, *args): pass
            def subheader(self, *args): pass
            def caption(self, *args): pass
            def selectbox(self, *args): return "Groq"
            def text_input(self, *args, **kwargs): return ""
            def checkbox(self, *args, **kwargs): return False
            def chat_input(self, *args): return "hi"
            def audio_input(self, *args, **kwargs): return None
            def chat_message(self, *args): return self
            def markdown(self, text): self.output.append(text)

            def error(self, text):
                raise AssertionError(text)

        def forbidden(*args, **kwargs):
            self.fail("A greeting must not initialize a provider or search database")

        st = FakeStreamlit()
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        namespace = {"__name__": "pseudo_app", "logging": logging, "os": os,
                     "st": st, "PROJECT_ROOT": ROOT, "Path": Path,
                     "TemporaryDirectory": TemporaryDirectory, "AIMessage": AIMessage,
                     "HumanMessage": HumanMessage, "get_session_workflow": forbidden,
                     "rag": forbidden, "get_greeting_response": load_retrieval()["get_greeting_response"]}
        exec(compile(tree, "app.py", "exec"), namespace)
        self.assertEqual(len(st.session_state["chat_history"]), 3)
        self.assertTrue(st.session_state["chat_history"][-1].content.startswith("Hi!"))
        self.assertTrue(st.output[-1].startswith("Hi!"))

    def test_repository_tweets_search_without_external_dependencies(self):
        os.environ["RETRIEVAL_MODE"] = "keyword"
        namespace = load_retrieval()
        context, urls = namespace["search_tweets"]("Chelsea", k=3)
        self.assertGreater(len(context), 0)
        self.assertLessEqual(len(context), 3)
        self.assertEqual(len(context), len(urls))
        for text, url in zip(context, urls):
            self.assertIn("chelsea", text.casefold())
            self.assertIn(url, text)
            self.assertTrue(url.startswith("https://"))

    def test_hybrid_vector_search_is_default_and_initializes_before_search(self):
        namespace = load_retrieval()
        for value in (None, "", "  ", "HYBRID"):
            if value is None:
                os.environ.pop("RETRIEVAL_MODE", None)
            else:
                os.environ["RETRIEVAL_MODE"] = value
            self.assertEqual(namespace["get_retrieval_mode"](), "hybrid")
        calls = []
        database = ModuleType("init_lancedb")
        database.initialize_database = lambda: calls.append("initialize")
        vectordb = ModuleType("src.vectordb")
        def search(query, k):
            calls.append(("hybrid", query, k))
            return ["vector context"], ["https://example.com/tweet"]
        vectordb.lancedb_hybrid_search = search
        with patch.dict("sys.modules", {"init_lancedb": database, "src.vectordb": vectordb}):
            self.assertEqual(namespace["search_tweets"]("Chelsea", k=2),
                             (["vector context"], ["https://example.com/tweet"]))
        self.assertEqual(calls, ["initialize", ("hybrid", "Chelsea", 2)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
