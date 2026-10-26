import unittest

import pytest
pytest.importorskip("qwen_tts", reason="optional native Qwen GPU backend environment")
import torch
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

from simulinfer.tts_backends import qwen as service


class QwenHistoryTests(unittest.TestCase):
    def setUp(self):
        self.original = service.VOICES.get(service.DEFAULT_VOICE)
        self.anchor = VoiceClonePromptItem(
            ref_code=torch.zeros((3, 16), dtype=torch.long),
            ref_spk_embedding=torch.ones(1024),
            x_vector_only_mode=False, icl_mode=True, ref_text="reference",
        )
        service.VOICES[service.DEFAULT_VOICE] = {"prompt": self.anchor}

    def tearDown(self):
        if self.original is None:
            service.VOICES.pop(service.DEFAULT_VOICE, None)
        else:
            service.VOICES[service.DEFAULT_VOICE] = self.original

    def test_generated_speech_and_transcript_become_next_prompt(self):
        session = service.QwenSession()
        codes = torch.full((5, 16), 123, dtype=torch.long)
        session.commit("first sentence", codes, "en")
        prompt = session.prompt("en")
        self.assertEqual(prompt.ref_text, "first sentence")
        self.assertTrue(torch.equal(prompt.ref_code, codes))
        self.assertIs(prompt.ref_spk_embedding, self.anchor.ref_spk_embedding)
        self.assertTrue(prompt.icl_mode)
        self.assertFalse(prompt.x_vector_only_mode)

    def test_window_removes_complete_pairs(self):
        session = service.QwenSession(window_s=1)
        session.commit("first", torch.ones((7, 16), dtype=torch.long), "en")
        second = torch.full((7, 16), 2, dtype=torch.long)
        session.commit("second", second, "en")
        self.assertEqual(session.prompt("en").ref_text, "second")
        self.assertTrue(torch.equal(session.prompt("en").ref_code, second))

    def test_language_change_restores_original_reference(self):
        session = service.QwenSession()
        session.commit("previous English", torch.ones((5, 16), dtype=torch.long), "en")
        self.assertIs(session.prompt("zh"), self.anchor)
        self.assertEqual(session.history, [])

    def test_refresh_same_voice_preserves_generated_speech(self):
        session = service.QwenSession()
        codes = torch.ones((5, 16), dtype=torch.long)
        session.commit("previous words", codes, "en")
        updated = VoiceClonePromptItem(
            ref_code=None, ref_spk_embedding=torch.full((1024,), 2.0),
            x_vector_only_mode=True, icl_mode=False, ref_text=None,
        )
        service.VOICES[service.DEFAULT_VOICE] = {"prompt": updated}
        self.assertTrue(session.refresh_voice())
        prompt = session.prompt("en")
        self.assertEqual(prompt.ref_text, "previous words")
        self.assertTrue(torch.equal(prompt.ref_code, codes))
        self.assertIs(prompt.ref_spk_embedding, updated.ref_spk_embedding)
        self.assertEqual(session.steps, 1)
        self.assertFalse(session.refresh_voice())

    def test_switch_to_cloned_voice_preserves_history(self):
        voice = "unit_live_voice"
        updated = VoiceClonePromptItem(
            ref_code=None, ref_spk_embedding=torch.full((1024,), 3.0),
            x_vector_only_mode=True, icl_mode=False, ref_text=None,
        )
        service.VOICES[voice] = {"prompt": updated}
        try:
            session = service.QwenSession()
            codes = torch.ones((5, 16), dtype=torch.long)
            session.commit("previous words", codes, "en")
            self.assertTrue(session.refresh_voice(voice))
            prompt = session.prompt("en")
            self.assertEqual(prompt.ref_text, "previous words")
            self.assertTrue(torch.equal(prompt.ref_code, codes))
            self.assertIs(prompt.ref_spk_embedding, updated.ref_spk_embedding)
            self.assertEqual(session.voice, voice)
        finally:
            service.VOICES.pop(voice)

    def test_long_chunk_remains_an_aligned_reference(self):
        session = service.QwenSession(window_s=1)
        codes = torch.ones((25, 16), dtype=torch.long)
        session.commit("long chunk", codes, "en")
        self.assertEqual(session.prompt("en").ref_text, "long chunk")
        self.assertTrue(torch.equal(session.prompt("en").ref_code, codes))
        session.commit("next", torch.ones((5, 16), dtype=torch.long), "en")
        self.assertEqual(session.prompt("en").ref_text, "next")

    def test_oversized_history_returns_to_original_anchor(self):
        session = service.QwenSession()
        session.commit("too long", torch.ones((361, 16), dtype=torch.long), "en")
        self.assertEqual(session.history, [])
        self.assertIs(session.prompt("en"), self.anchor)

    def test_completion_requires_actual_codec_eos(self):
        service.require_codec_eos(torch.tensor([[1, 2, 2150]]), 2150)
        with self.assertRaisesRegex(RuntimeError, "without codec EOS"):
            service.require_codec_eos(torch.tensor([[1, 2, 3]]), 2150)


if __name__ == "__main__":
    unittest.main()
