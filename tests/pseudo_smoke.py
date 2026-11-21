"""Dependency-free checks of the actual functions with controlled substitutes.

Run: python tests/pseudo_smoke.py
These checks do not launch Streamlit, embed tweets, or call external services.
"""

import ast
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    return namespace


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
            "lancedb_hybrid_search": lambda query: (["tweet one", "tweet two"], ["url"]),
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
                    "lancedb_hybrid_search": fail if stage == "retrieval" else lambda query: (["tweet"], ["url"]),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
