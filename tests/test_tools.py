"""Tests for the agent-facing boundary.

    python -m unittest discover tests

Two properties matter more than the rest here, because this is the last layer before
the model and anything lost at it is lost for good:

  * every culture mentioned anywhere in a response has its attribution attached, and
  * no path we are forbidden to serve appears in a payload at all.

Both are asserted over every tool and every culture rather than on known cases.
"""

from __future__ import annotations

import json
import sqlite3
import unittest

from skylore import paths, tools

DATABASE = paths.DATABASE


@unittest.skipUnless(DATABASE.exists(), "corpus.db not built")
class Boundary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db = tools.connect(DATABASE)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def responses(self) -> list[tuple[str, dict]]:
        """One representative call per tool, in a couple of languages."""
        return [
            ("find_cultures", tools.find_cultures(self.db)),
            ("find_cultures/ru", tools.find_cultures(self.db, lang="ru")),
            ("find_cultures/region", tools.find_cultures(self.db, region="Oceania")),
            ("get_culture_article", tools.get_culture_article(self.db, culture="lokono")),
            ("get_culture_article/jms",
             tools.get_culture_article(self.db, culture="japanese_moon_stations")),
            ("lookup_star", tools.lookup_star(self.db, query="Aldebaran", lang="ru")),
            ("lookup_star/hip", tools.lookup_star(self.db, query="HIP 21421")),
            ("find_constellation",
             tools.find_constellation(self.db, query="毕宿", lang="zh-Hans")),
            ("find_constellation/ru",
             tools.find_constellation(self.db, query="Сетка", lang="ru")),
        ]


class Connection(Boundary):

    def test_the_connection_is_read_only(self):
        # Nothing in the query layer writes, so nothing may.
        with self.assertRaises(sqlite3.OperationalError):
            self.db.execute("DELETE FROM cultures")


class Serialisation(Boundary):

    def test_every_response_is_json(self):
        for label, payload in self.responses():
            json.dumps(payload, ensure_ascii=False)  # must not raise
            self.assertIsInstance(payload, dict, label)

    def test_native_names_are_never_filed_as_translatable_text(self):
        payload = tools.find_constellation(self.db, query="毕宿", lang="zh-Hans")
        nameset = payload["constellations"][0]["names"]
        self.assertEqual(nameset["native"], "毕宿")
        self.assertEqual(nameset["pronounce"], "Bìxiù")
        # `meanings` is the language-keyed dictionary; native/pronounce are not in it.
        self.assertNotIn("native", nameset["meanings"])
        self.assertNotIn(None, nameset["meanings"])

    def test_the_whole_name_dictionary_survives(self):
        payload = tools.lookup_star(self.db, query="Aldebaran")
        bugis = next(n for n in payload["stars"][0]["named_by"]
                     if n["culture"] == "bugis")
        self.assertEqual(set(bugis["meanings"]), {"en", "ru", "es", "zh_CN"})

    def test_naming_and_drawing_stay_separate(self):
        star = tools.lookup_star(self.db, query="HIP 21421")["stars"][0]
        named = {n["culture"] for n in star["named_by"] if n["culture"]}
        drawn = {c["culture"] for c in star["drawn_into"]}
        self.assertTrue(named and drawn)
        self.assertTrue(drawn - named, "expected a culture that draws but does not name")

    def test_citations_are_resolved_not_left_as_markers(self):
        payload = tools.get_culture_article(self.db, culture="anutan")
        cited = [s for s in payload["sections"] if "references" in s]
        self.assertTrue(cited)
        for section in cited:
            for number, text in section["references"].items():
                self.assertIn(f"[#{number}]", section["text"])
                self.assertTrue(text.strip())


class Provenance(Boundary):
    """The resolver's whole output is provenance; it must not die at the boundary."""

    def test_a_fallback_is_flagged(self):
        star = tools.lookup_star(self.db, query="Aldebaran", lang="ru")["stars"][0]
        chinese = next(n for n in star["named_by"] if n["culture"] == "chinese")
        self.assertTrue(chinese["preferred"]["is_fallback"])
        self.assertEqual(chinese["preferred"]["requested"], "ru")
        self.assertEqual(chinese["preferred"]["lang"], "en")

    def test_a_genuine_translation_is_not_flagged(self):
        star = tools.lookup_star(self.db, query="Aldebaran", lang="ru")["stars"][0]
        bugis = next(n for n in star["named_by"] if n["culture"] == "bugis")
        self.assertNotIn("is_fallback", bugis["preferred"])
        self.assertEqual(bugis["preferred"]["lang"], "ru")

    def test_both_cases_occur_in_one_response(self):
        # If only one ever occurred, the two tests above would be checking a constant.
        star = tools.lookup_star(self.db, query="Aldebaran", lang="ru")["stars"][0]
        flags = {n["preferred"].get("is_fallback", False)
                 for n in star["named_by"] if "preferred" in n}
        self.assertEqual(flags, {True, False})

    def test_a_chinese_request_gets_chinese_without_a_fallback_flag(self):
        payload = tools.find_constellation(self.db, query="毕宿", lang="zh-Hans")
        preferred = payload["constellations"][0]["names"]["preferred"]
        self.assertEqual(preferred["value"], "毕宿")
        self.assertNotIn("is_fallback", preferred)

    def test_prose_reports_the_language_it_is_actually_in(self):
        payload = tools.get_culture_article(self.db, culture="lokono", lang="ru")
        self.assertTrue(all(s["lang"] == "en" for s in payload["sections"]))


class Licensing(Boundary):

    def test_attribution_is_attached_for_every_culture_mentioned(self):
        for label, payload in self.responses():
            mentioned = tools._cultures_mentioned(
                self.db, {k: v for k, v in payload.items() if k != "sources"})
            self.assertTrue(mentioned, label)
            self.assertLessEqual(mentioned, set(payload.get("sources", {})), label)
            for culture_id in mentioned:
                self.assertTrue(payload["sources"][culture_id]["attribution"].strip(),
                                f"{label}: {culture_id}")

    def test_constellation_ids_are_not_mistaken_for_culture_ids(self):
        # "CON western Tau" sits under an `id` key too. Keying the invariant check on
        # key names alone would collect it and make the check meaningless.
        star = tools.lookup_star(self.db, query="Aldebaran")
        raw = tools._identifiers({k: v for k, v in star.items() if k != "sources"})
        resolved = tools._cultures_mentioned(self.db, star)
        self.assertTrue(any(i.startswith("CON ") for i in raw), "figures carry ids")
        self.assertFalse(any(i.startswith("CON ") for i in resolved))
        self.assertLessEqual(resolved, tools.known_cultures(self.db))

    def test_artwork_terms_travel_beside_prose_terms(self):
        # Free Art License is copyleft, so anything showing those images owes LAL
        # terms rather than only the prose CC-BY-SA.
        payload = tools.find_cultures(self.db)
        for culture_id, source in payload["sources"].items():
            self.assertIn("text_licenses", source, culture_id)
            self.assertIn("image_licenses", source, culture_id)
            self.assertIn("images_usable", source, culture_id)

    def test_no_excluded_asset_path_appears_in_any_payload(self):
        excluded = [row[0] for row in
                    self.db.execute("SELECT path FROM excluded_assets")]
        self.assertTrue(excluded, "the corpus has at least one excluded asset")
        for label, payload in self.responses():
            serialised = json.dumps(payload, ensure_ascii=False)
            for path in excluded:
                self.assertNotIn(path, serialised, f"{label} leaks {path}")

    def test_removal_is_reported_without_naming_the_file(self):
        payload = tools.get_culture_article(
            self.db, culture="japanese_moon_stations")
        self.assertEqual(payload["omitted"]["images"], 1)
        self.assertNotIn("chart.webp", json.dumps(payload))

    def test_a_full_sweep_of_every_culture_leaks_nothing(self):
        excluded = [row[0] for row in
                    self.db.execute("SELECT path FROM excluded_assets")]
        for card in tools.find_cultures(self.db)["cultures"]:
            serialised = json.dumps(
                tools.get_culture_article(self.db, culture=card["id"]),
                ensure_ascii=False)
            for path in excluded:
                self.assertNotIn(path, serialised, card["id"])


class Dispatch(Boundary):

    def test_every_schema_names_a_real_tool(self):
        self.assertEqual({s["name"] for s in tools.SCHEMAS}, set(tools.TOOLS))

    def test_schemas_are_well_formed(self):
        for schema in tools.SCHEMAS:
            self.assertTrue(schema["description"].strip(), schema["name"])
            self.assertEqual(schema["input_schema"]["type"], "object")
            for prop in schema["input_schema"]["properties"].values():
                self.assertIn("type", prop)

    def test_schema_parameters_match_the_functions(self):
        import inspect
        for schema in tools.SCHEMAS:
            signature = inspect.signature(tools.TOOLS[schema["name"]])
            declared = set(schema["input_schema"]["properties"])
            accepted = set(signature.parameters) - {"connection"}
            self.assertLessEqual(declared, accepted, schema["name"])

    def test_required_parameters_have_no_default(self):
        import inspect
        for schema in tools.SCHEMAS:
            signature = inspect.signature(tools.TOOLS[schema["name"]])
            for name in schema["input_schema"].get("required", []):
                self.assertIs(signature.parameters[name].default,
                              inspect.Parameter.empty, f"{schema['name']}.{name}")

    def test_mechanism_is_not_exposed_as_a_tool(self):
        # `names.search` reads the database but is deliberately absent: a tool named
        # after mechanism makes the model choose between exact and fuzzy matching.
        self.assertNotIn("search", tools.TOOLS)

    def test_call_dispatches(self):
        result = tools.call(self.db, "find_constellation", {"query": "Orion"})
        self.assertIn("constellations", result)

    def test_an_unknown_tool_answers_rather_than_raising(self):
        result = tools.call(self.db, "nope", {})
        self.assertIn("error", result)
        self.assertEqual(set(result["available"]), set(tools.TOOLS))

    def test_bad_arguments_answer_rather_than_raising(self):
        result = tools.call(self.db, "lookup_star", {"bogus": 1})
        self.assertIn("error", result)

    def test_an_unknown_culture_answers_with_a_way_forward(self):
        result = tools.call(self.db, "get_culture_article", {"culture": "nope"})
        self.assertIn("error", result)
        self.assertIn("find_cultures", result["hint"])


class Budget(Boundary):

    def test_the_catalogue_still_fits_in_a_prompt(self):
        # The premise of reading the catalogue instead of searching it. Measured at
        # ~25k characters, of which attribution is roughly half.
        size = len(json.dumps(tools.find_cultures(self.db), ensure_ascii=False))
        self.assertLess(size, 60_000, "catalogue has outgrown being read whole")

    def test_a_star_lookup_stays_bounded(self):
        # Aldebaran is the worst case: 27 figures across 26 cultures.
        size = len(json.dumps(tools.lookup_star(self.db, query="Aldebaran"),
                              ensure_ascii=False))
        self.assertLess(size, 60_000)


if __name__ == "__main__":
    unittest.main()
