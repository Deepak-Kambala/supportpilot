import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src import assistant, knowledge_base


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        # Deterministic vectors make this a local retrieval smoke test without downloads.
        return np.asarray([[1.0, 0.0] if "invite" in t.lower() else [0.0, 1.0] for t in texts], dtype="float32")


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, prompt, **kwargs):
        return FakeBatch(input_ids=torch.tensor([[1, 2]]))

    def decode(self, tokens, **kwargs):
        return "decoded model completion"


class FakeModel:
    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


class SupportPilotCoreTests(unittest.TestCase):
    def test_generation_returns_decoded_model_completion(self):
        raw, _ = assistant.generate_from_messages(
            [{"role": "user", "content": "password reset test"}],
            FakeTokenizer(), FakeModel(), "cpu",
        )
        self.assertEqual(raw, "decoded model completion")

    def test_demo_document_chunks_and_retrieves(self):
        knowledge_base.get_embedder.cache_clear()
        with patch.object(knowledge_base, "get_embedder", return_value=FakeEmbedder()):
            path = "data/demo_docs/acme_faq.md"
            docs = [{"filename": "acme_faq.md", "text": knowledge_base.read_document(
                path, Path(path).read_bytes()
            )}]
            index, chunks = knowledge_base.build_index(docs)
            results = knowledge_base.retrieve("how to invite a teammate", index, chunks)
        self.assertTrue(chunks)
        self.assertEqual(results[0]["source"], "acme_faq.md")
        self.assertGreaterEqual(results[0]["score"], 0.99)
        invite_chunk = next(c["text"] for c in chunks if "invite member" in c["text"].lower())
        self.assertNotIn("billing", invite_chunk.lower())

    def run_reply(self, question, passages, model_json):
        with patch.object(assistant, "generate_from_messages", return_value=(model_json, 0.01)):
            return assistant.generate_reply(question, passages, object(), object(), "cpu")

    def test_insufficient_evidence_uses_safe_human_fallback(self):
        result = self.run_reply("How do I export data?", [{"text": "irrelevant", "score": 0.1}], "{}")
        self.assertTrue(result["needs_human"])
        self.assertFalse(result["valid_json"])
        self.assertIn("No documentation", result["reason"])

    def test_billing_category_requires_human_review(self):
        raw = '{"category":"billing","priority":"low","needs_human":false,"draft_reply":"Check billing settings.","reason":"Documentation found."}'
        result = self.run_reply("Why was I charged twice?", [{"text": "Billing guide", "score": 0.8}], raw)
        self.assertTrue(result["needs_human"])

    def test_billing_terms_require_review_even_if_model_misclassifies(self):
        raw = '{"category":"other","priority":"low","needs_human":false,"draft_reply":"Check the payment page.","reason":"Documentation found."}'
        result = self.run_reply("I was charged twice", [{"text": "Billing guide", "score": 0.8}], raw)
        self.assertTrue(result["needs_human"])

    def test_security_message_requires_high_priority_human_review(self):
        raw = '{"category":"account_access","priority":"low","needs_human":false,"draft_reply":"Reset your password.","reason":"Documentation found."}'
        result = self.run_reply("I think someone else logged into my account", [{"text": "Security guide", "score": 0.8}], raw)
        self.assertTrue(result["needs_human"])
        self.assertEqual(result["priority"], "high")


if __name__ == "__main__":
    unittest.main()
