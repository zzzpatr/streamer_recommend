import unittest
from types import SimpleNamespace

from web_enrichment import combine_semantic_query, expand_web_query


class _FakeResponse:
    output_parsed = SimpleNamespace(
        entity="芙莉蓮",
        expanded_query="冷靜、慢熟、溫柔、療癒陪伴感",
        explanation="以角色特質補充原始語意。",
    )

    def model_dump(self):
        return {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {
                                "title": "Source A",
                                "url": "https://example.com/a",
                            },
                            {
                                "title": "Duplicate",
                                "url": "https://example.com/a",
                            },
                            {
                                "title": "Unsafe",
                                "url": "javascript:alert(1)",
                            },
                        ]
                    },
                }
            ]
        }


class _FakeResponses:
    def __init__(self):
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return _FakeResponse()


class WebEnrichmentTests(unittest.TestCase):
    def test_search_is_required_and_sources_are_sanitized(self):
        responses = _FakeResponses()
        client = SimpleNamespace(responses=responses)

        result = expand_web_query(client, "test-model", "  芙莉蓮   性格 ")

        self.assertEqual(responses.request["tools"], [{"type": "web_search"}])
        self.assertEqual(responses.request["tool_choice"], "required")
        self.assertEqual(result.lookup_query, "芙莉蓮 性格")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].url, "https://example.com/a")

    def test_expansion_is_only_appended_to_semantic_query(self):
        responses = _FakeResponses()
        enrichment = expand_web_query(
            SimpleNamespace(responses=responses),
            "test-model",
            "芙莉蓮",
        )

        combined = combine_semantic_query("像芙莉蓮", enrichment)

        self.assertIn("像芙莉蓮", combined)
        self.assertIn("外部概念特徵", combined)
        self.assertIn("療癒陪伴感", combined)


if __name__ == "__main__":
    unittest.main()
