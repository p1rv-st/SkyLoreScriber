"""Tests for the agent layer, run without a network or an API key.

    python -m unittest discover tests

The whole file skips when `pydantic-ai` is absent, because the project's core installs
with no dependencies at all and `python -m unittest discover tests` has to keep passing
in that state. Install it with `uv sync --extra agent`.

What is worth asserting here is narrow. The tools themselves are tested in
`test_tools.py`; this layer only wires them up, so the tests guard the three ways that
wiring can go wrong silently:

  * a description the model never receives -- the schemas are the reviewed text, and a
    parameter renamed in a wrapper drops its description without any error,
  * a tool that raises instead of answering, which in this layer means the sqlite
    thread-affinity trap the async wrappers exist to avoid, and
  * attribution lost between `skylore.tools` and the model.
"""

from __future__ import annotations

import inspect
import os
import unittest
import unittest.mock

try:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolReturnPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.models.test import TestModel
except ImportError:  # pragma: no cover - exercised by installing without the extra
    FunctionModel = None

from skylore import paths, tools

DATABASE = paths.DATABASE


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
class Wiring(unittest.TestCase):
    """The definitions the model receives, checked against `tools.SCHEMAS`."""

    def setUp(self):
        from skylore.agent import loop as agent
        self.agent = agent

    def test_every_tool_is_registered_once(self):
        self.assertEqual(sorted(self.agent.CORPUS_TOOLS), sorted(tools.TOOLS))
        names = [tool.name for tool in self.agent.toolset(internet=False)]
        self.assertEqual(sorted(names), sorted(tools.TOOLS))
        self.assertEqual(len(names), len(set(names)))

    def test_the_web_tool_is_absent_unless_asked_for(self):
        """Not registered rather than registered and refused: a tool the model cannot
        see is one it cannot misuse."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            names = {tool.name for tool in self.agent.toolset()}
        self.assertNotIn("search_web", names)

        names = {tool.name for tool in self.agent.toolset(internet=True)}
        self.assertEqual(names, set(tools.TOOLS) | {"search_web"})

    def test_wrapper_signature_matches_the_schema(self):
        """Parameter names come from the signature and descriptions from the schema, so
        a drift between them loses a description with no error anywhere."""
        for name, schema in self.agent.SCHEMAS.items():
            with self.subTest(tool=name):
                wrapper = self.agent.WRAPPERS[name]
                parameters = set(inspect.signature(wrapper).parameters) - {"ctx"}
                self.assertEqual(parameters, set(schema["input_schema"]["properties"]))

    def test_wrappers_are_async(self):
        """Not a style rule. A sync tool function runs in a worker thread, and a sqlite
        connection may only be used in the thread that opened it."""
        for name, wrapper in self.agent.WRAPPERS.items():
            with self.subTest(tool=name):
                self.assertTrue(inspect.iscoroutinefunction(wrapper))

    def test_required_arguments_survive(self):
        for name, schema in self.agent.SCHEMAS.items():
            with self.subTest(tool=name):
                required = set(schema["input_schema"].get("required", []))
                signature = inspect.signature(self.agent.WRAPPERS[name])
                for argument in required:
                    parameter = signature.parameters[argument]
                    self.assertIs(parameter.default, inspect.Parameter.empty)


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class WhatTheModelSees(unittest.TestCase):
    """One run against a stub model, inspecting the request rather than answering it."""

    def definitions(self, lang: str = "ru"):
        """Built with the web tool on, so all seven descriptions are checked -- the one
        that is optional at runtime is not optional to describe properly."""
        from skylore.agent import loop as agent

        captured = {}

        def stub(messages, info: AgentInfo):
            captured["tools"] = {tool.name: tool for tool in info.function_tools}
            captured["instructions"] = "\n".join(
                part.content for message in messages
                for part in getattr(message, "parts", ())
                if type(part).__name__ == "SystemPromptPart"
            ) + (messages[0].instructions or "")
            return ModelResponse(parts=[TextPart("stubbed")])

        connection = tools.connect(DATABASE)
        try:
            with unittest.mock.patch.dict(os.environ,
                                          {"TAVILY_API_KEY": "tvly-test"}):
                # Every optional tool on, because this class sweeps *every* schema the
                # model can be shown -- a tool left off here is a description nobody
                # checks arrived.
                agent.build(FunctionModel(stub), internet=True,
                            images=True).run_sync(
                    "test", deps=agent.Deps(connection, lang=lang))
        finally:
            connection.close()
        return captured

    def test_descriptions_are_the_schema_text(self):
        from skylore.agent import loop as agent
        definitions = self.definitions()["tools"]
        for name, schema in agent.SCHEMAS.items():
            with self.subTest(tool=name):
                self.assertEqual(definitions[name].description, schema["description"])

    def test_parameter_descriptions_reach_the_model(self):
        """Every description written in a schema arrives; none is quietly dropped."""
        from skylore.agent import loop as agent
        definitions = self.definitions()["tools"]
        for name, schema in agent.SCHEMAS.items():
            for argument, spec in schema["input_schema"]["properties"].items():
                if not spec.get("description"):
                    continue
                with self.subTest(tool=name, argument=argument):
                    arrived = definitions[name].parameters_json_schema["properties"]
                    self.assertIn("description", arrived[argument])

    def test_the_users_language_is_stated(self):
        instructions = self.definitions(lang="zh-Hans")["instructions"]
        self.assertIn("zh-Hans", instructions)
        # The licensing obligation is the one thing the answer must carry that the
        # corpus cannot enforce for itself.
        self.assertIn("Attribution is not optional", instructions)


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class EveryToolRuns(unittest.TestCase):
    """`TestModel` calls all six with schema-valid arguments, which is the cheapest
    check that no tool raises when the agent -- rather than a test -- invokes it."""

    @classmethod
    def setUpClass(cls):
        from skylore.agent import loop as agent
        cls.agent_module = agent
        cls.connection = tools.connect(DATABASE)
        cls.result = agent.build(TestModel()).run_sync(
            "test", deps=agent.Deps(cls.connection))
        cls.returns = [part for message in cls.result.all_messages()
                       for part in getattr(message, "parts", ())
                       if isinstance(part, ToolReturnPart)]

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def test_all_six_were_called(self):
        called = {part.tool_name for part in self.returns}
        self.assertEqual(called, set(tools.TOOLS))

    def test_each_answered_with_a_payload(self):
        for part in self.returns:
            with self.subTest(tool=part.tool_name):
                self.assertIsInstance(part.content, dict)

    def test_the_trajectory_records_every_call(self):
        calls = self.agent_module.trajectory(self.result.all_messages())
        self.assertEqual({call.tool for call in calls}, set(tools.TOOLS))
        for call in calls:
            self.assertIsInstance(call.arguments, dict)


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class AttributionSurvivesTheWrapper(unittest.TestCase):
    """`_with_sources` runs below this layer; this asserts the wrapper does not flatten
    it away on the path to the model."""

    def test_sources_reach_the_model(self):
        from skylore.agent import loop as agent

        def call_one(messages, info: AgentInfo):
            if len(messages) == 1:
                from pydantic_ai.messages import ToolCallPart
                return ModelResponse(parts=[ToolCallPart(
                    "lookup_star", {"query": "Aldebaran", "lang": "en"})])
            return ModelResponse(parts=[TextPart("stubbed")])

        connection = tools.connect(DATABASE)
        try:
            result = agent.build(FunctionModel(call_one)).run_sync(
                "who names Aldebaran?", deps=agent.Deps(connection))
        finally:
            connection.close()

        payloads = [part.content for message in result.all_messages()
                    for part in getattr(message, "parts", ())
                    if isinstance(part, ToolReturnPart)]
        self.assertTrue(payloads)
        self.assertIn("sources", payloads[0])
        self.assertTrue(payloads[0]["sources"])
        for terms in payloads[0]["sources"].values():
            self.assertTrue(terms["attribution"])


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class PictureLimit(unittest.TestCase):
    """A model that wants a gallery gets two pictures and a reason.

    The schema asks for restraint, and asking was measured: on "show me what they look
    like" about five figures, the model called the tool six times. So the wrapper counts.
    """

    def run_greedy(self, wanted: int):
        from pydantic_ai.messages import ToolCallPart
        from skylore.agent import loop as agent

        def greedy(messages, info: AgentInfo):
            returned = [part for message in messages
                        for part in getattr(message, "parts", ())
                        if isinstance(part, ToolReturnPart)]
            if len(returned) < wanted:
                return ModelResponse(parts=[ToolCallPart(
                    "show_constellation_image", {"constellation": "CON aztec 001"})])
            return ModelResponse(parts=[TextPart("done")])

        connection = tools.connect(DATABASE)
        try:
            result = agent.build(FunctionModel(greedy), images=True).run_sync(
                "show me everything", deps=agent.Deps(connection))
        finally:
            connection.close()
        return [part.content for message in result.all_messages()
                for part in getattr(message, "parts", ())
                if isinstance(part, ToolReturnPart)]

    def test_the_first_two_are_shown_and_the_rest_refused(self):
        from skylore.agent import loop as agent

        payloads = self.run_greedy(5)
        shown = [p for p in payloads if p.get("image")]
        refused = [p for p in payloads if "refused" in p]
        self.assertEqual(len(shown), agent.IMAGE_LIMIT)
        self.assertTrue(refused)
        self.assertIn("limit", refused[0]["refused"]["reason"])

    def test_the_refusal_says_what_to_do_instead(self):
        """A refusal with no alternative reads as an invitation to try again."""
        refused = next(p for p in self.run_greedy(4) if "refused" in p)
        self.assertIn("words", refused["refused"]["hint"])


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
class Trajectory(unittest.TestCase):
    """Arguments arrive as a dict or as a JSON string depending on the provider."""

    def message(self, arguments):
        from pydantic_ai.messages import ToolCallPart
        return ModelResponse(parts=[ToolCallPart("lookup_star", arguments)])

    def test_json_string_arguments_are_parsed(self):
        from skylore.agent import loop as agent
        calls = agent.trajectory([self.message('{"query": "Sirius"}')])
        self.assertEqual(calls[0].arguments, {"query": "Sirius"})

    def test_unparseable_arguments_are_kept(self):
        """A malformed call is exactly what a trajectory judge needs to see, so it is
        recorded rather than dropped."""
        from skylore.agent import loop as agent
        calls = agent.trajectory([self.message('{"query": ')])
        self.assertEqual(calls[0].tool, "lookup_star")
        self.assertIn("__unparsed__", calls[0].arguments)


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
class WebSearch(unittest.TestCase):
    """`skylore.web`, exercised without a network or a key.

    Only the shape can go wrong here, and the shape is what carries the guarantee that a
    web result is never mistaken for corpus material.
    """

    def test_off_by_default(self):
        from skylore.agent import web
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(web.enabled())

    def test_the_environment_turns_it_on(self):
        from skylore.agent import web
        for value in ("true", "TRUE", "1", "yes", "on"):
            with self.subTest(value=value):
                with unittest.mock.patch.dict(os.environ,
                                              {"INTERNET_SEARCH": value}):
                    self.assertTrue(web.enabled())
        for value in ("false", "0", "no", ""):
            with self.subTest(value=value):
                with unittest.mock.patch.dict(os.environ,
                                              {"INTERNET_SEARCH": value}):
                    self.assertFalse(web.enabled())

    def test_an_explicit_argument_wins(self):
        from skylore.agent import web
        with unittest.mock.patch.dict(os.environ, {"INTERNET_SEARCH": "true"}):
            self.assertFalse(web.enabled(False))
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(web.enabled(True))

    def test_a_missing_key_is_refused_by_name(self):
        from skylore.agent import web
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as caught:
                web.api_key()
        self.assertIn("TAVILY_API_KEY", str(caught.exception))

    def test_the_agent_refuses_to_build_without_a_key(self):
        """Asked-for-and-broken is worse than off, so this fails at build time rather
        than returning nothing from every search for a whole run."""
        from skylore.agent import loop as agent
        with unittest.mock.patch.dict(os.environ, {"INTERNET_SEARCH": "true"},
                                      clear=True):
            with self.assertRaises(ValueError):
                agent.build(TestModel())

    def run_search(self, response):
        import asyncio
        from skylore.agent import web

        class Response:
            def __init__(self, body):
                self._body = body

            def raise_for_status(self):
                return None

            def json(self):
                return self._body

        class Client:
            def __init__(self, body):
                self._body = body
                self.request = None

            async def post(self, url, json, headers):
                self.request = {"url": url, "json": json, "headers": headers}
                if isinstance(self._body, Exception):
                    raise self._body
                return Response(self._body)

            async def aclose(self):
                return None

        client = Client(response)
        payload = asyncio.run(
            web.search("Aldebaran transliteration", key="tvly-test", client=client))
        return payload, client

    def test_results_are_labelled_external(self):
        payload, _ = self.run_search({"results": [
            {"title": "A", "url": "https://example.org/a", "content": "text",
             "score": 0.9}]})
        self.assertTrue(payload["external"])
        self.assertIn("licence", payload)
        self.assertEqual(payload["results"][0]["url"], "https://example.org/a")
        self.assertEqual(payload["results"][0]["text"], "text")
        # No sky culture is credited, and nothing here may be read as corpus material.
        self.assertNotIn("sources", payload)

    def test_a_synthesised_answer_is_never_requested(self):
        """This layer hands the model sources. A second model's synthesis in between is
        a hop whose provenance nobody can check."""
        _, client = self.run_search({"results": []})
        self.assertFalse(client.request["json"]["include_answer"])
        self.assertFalse(client.request["json"]["include_raw_content"])
        self.assertEqual(client.request["headers"]["Authorization"], "Bearer tvly-test")

    def test_max_results_is_clamped(self):
        import asyncio
        from skylore.agent import web
        _, client = self.run_search({"results": []})
        self.assertEqual(client.request["json"]["max_results"], web.MAX_RESULTS)

        class Client:
            async def post(self, url, json, headers):
                self.request = {"json": json}
                raise RuntimeError("stop here")

            async def aclose(self):
                return None

        client = Client()
        asyncio.run(web.search("q", max_results=99, key="k", client=client))
        self.assertEqual(client.request["json"]["max_results"], 10)

    def test_a_failed_search_answers_instead_of_raising(self):
        """The agent has a corpus to fall back on, so a network failure must not end the
        run -- but it has to be visible, or absence of evidence is reported as evidence
        of absence."""
        payload, _ = self.run_search(RuntimeError("connection refused"))
        self.assertIn("error", payload)
        self.assertIn("connection refused", payload["error"])
        self.assertTrue(payload["external"])
        self.assertIn("hint", payload)


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class WebInstructions(unittest.TestCase):

    def instructions(self, **environment):
        from skylore.agent import loop as agent
        captured = {}

        def stub(messages, info: AgentInfo):
            captured["text"] = messages[0].instructions or ""
            captured["tools"] = {tool.name for tool in info.function_tools}
            return ModelResponse(parts=[TextPart("stubbed")])

        connection = tools.connect(DATABASE)
        try:
            with unittest.mock.patch.dict(os.environ, environment, clear=True):
                agent.build(FunctionModel(stub)).run_sync(
                    "test", deps=agent.Deps(connection))
        finally:
            connection.close()
        return captured

    def test_no_web_instruction_when_the_tool_is_absent(self):
        """An instruction describing a tool the model cannot see invites it to explain
        what it would have done instead of doing what it can."""
        captured = self.instructions()
        self.assertNotIn("search_web", captured["tools"])
        self.assertNotIn("search_web", captured["text"])

    def test_the_web_instruction_arrives_with_the_tool(self):
        captured = self.instructions(INTERNET_SEARCH="true", TAVILY_API_KEY="tvly-test")
        self.assertIn("search_web", captured["tools"])
        self.assertIn("search_web", captured["text"])
        # The two rules that keep the corpus and the web apart in the answer.
        self.assertIn("never credit a sky culture", captured["text"])
        self.assertIn("corpus comes first", captured["text"])


@unittest.skipIf(FunctionModel is None, "pydantic-ai not installed (uv sync --extra agent)")
class Providers(unittest.TestCase):
    """No key needed: the naming rule is separate from building a client precisely so
    this can be tested."""

    def test_openrouter_qualifies_a_bare_model_id(self):
        from skylore.agent import loop as agent
        self.assertEqual(agent.qualify(agent.MODEL, "openrouter"),
                         f"openai/{agent.MODEL}")

    def test_openrouter_leaves_a_qualified_id_alone(self):
        from skylore.agent import loop as agent
        self.assertEqual(agent.qualify("anthropic/claude-x", "openrouter"),
                         "anthropic/claude-x")

    def test_openai_takes_the_id_as_written(self):
        from skylore.agent import loop as agent
        self.assertEqual(agent.qualify(agent.MODEL, "openai"), agent.MODEL)

    def test_openai_is_the_default(self):
        from skylore.agent import loop as agent
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(agent.provider_name(), "openai")

    def test_the_environment_selects_the_provider(self):
        from skylore.agent import loop as agent
        with unittest.mock.patch.dict(os.environ,
                                      {"SKYLORE_PROVIDER": "openrouter"}):
            self.assertEqual(agent.provider_name(), "openrouter")
        # An explicit argument still wins, which is what the CLI flag relies on.
        with unittest.mock.patch.dict(os.environ,
                                      {"SKYLORE_PROVIDER": "openrouter"}):
            self.assertEqual(agent.provider_name("openai"), "openai")

    def test_an_unknown_provider_is_refused(self):
        from skylore.agent import loop as agent
        with self.assertRaises(ValueError):
            agent.provider_name("hopeful")


if __name__ == "__main__":
    unittest.main()
