import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock, patch

from evaluate import load_test_cases
from generate_questions import generate_questions, make_question_plan, validate_draft
from ingestion import build_chunks, split_text_spans
from generator import RAGPipeline, EvidenceStatus, RetrievalSignal
from retriever import RetrievedChunk, NISTRetriever
import numpy as np


class WorkflowTests(unittest.TestCase):
    def test_withdrawal_metadata_survives_every_piece(self):
        parts = [c for c in build_chunks(strategy="chars") if c.control_id == "SA-12"]
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(c.is_withdrawn and c.withdrawal_notice.startswith("Withdrawn:") for c in parts))
        # Later pieces do not contain the original status text, so metadata is essential.
        self.assertNotIn("Withdrawn:", parts[-1].text)

    def pipeline(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._model = "test"
        pipeline._retriever = Mock()
        pipeline._retriever.retrieve.return_value = [
            RetrievedChunk("AC-2", "AC-2", "AC", "Account Management", "Manage accounts.", 0.9),
            RetrievedChunk("AC-3", "AC-3", "AC", "Access Enforcement", "Enforce access.", 0.8),
        ]
        return pipeline

    @patch("generator.ollama.chat")
    def test_organization_evidence_guard_cannot_be_overridden_by_high_scores(self, chat):
        pipeline = self.pipeline()
        for query in ("Does our organization implement MFA?", "Do we actually restrict changes?",
                      "우리 조직은 접근 통제를 실제로 이행하고 있나요?"):
            result = pipeline.query(query)
            self.assertEqual(result.retrieval_signal, RetrievalSignal.HIGH_SIMILARITY)
            self.assertEqual(result.evidence_status, EvidenceStatus.ORGANIZATION_EVIDENCE_REQUIRED)
        chat.assert_not_called()

    @patch("generator.ollama.chat")
    def test_semantic_status_query_and_malformed_classification_do_not_generate(self, chat):
        pipeline = self.pipeline()
        chat.return_value = {"message": {"content": '{"intent": "organization_status"}'}}
        result = pipeline.query("Is MFA enabled on these accounts?")
        self.assertEqual(result.evidence_status, EvidenceStatus.ORGANIZATION_EVIDENCE_REQUIRED)
        self.assertEqual(chat.call_count, 1)
        chat.reset_mock()
        chat.return_value = {"message": {"content": 'not JSON'}}
        result = pipeline.query("Access control?")
        self.assertEqual(result.evidence_status, EvidenceStatus.CLARIFICATION_REQUIRED)
        self.assertEqual(chat.call_count, 1)

    @patch("generator.ollama.chat")
    def test_requirements_answer_remains_unverified_even_with_valid_citations(self, chat):
        chat.side_effect = [
            {"message": {"content": '{"intent":"requirements"}'}},
            {"message": {"content": 'Manage accounts. [AC-2]'}},
        ]
        result = self.pipeline().query("How should we manage accounts?")
        self.assertEqual(result.evidence_status, EvidenceStatus.NEEDS_REVIEW)
        self.assertEqual(result.cited_control_ids, ["AC-2"])

    @patch("generator.ollama.chat")
    def test_unretrieved_and_malformed_citations_are_rejected(self, chat):
        for citation in ("SA-12", "AC-2(a)"):
            chat.side_effect = [
                {"message": {"content": '{"intent":"requirements"}'}},
                {"message": {"content": f'Manage accounts. [AC-2] Invalid claim. [{citation}]'}},
            ]
            result = self.pipeline().query("How should we manage accounts?")
            self.assertEqual(result.evidence_status, EvidenceStatus.INVALID_CITATIONS)
            self.assertNotIn("Invalid claim", result.answer)

    def test_withdrawn_hits_are_filtered_before_filling_top_k(self):
        retriever = NISTRetriever.__new__(NISTRetriever)
        retriever.model = Mock()
        retriever.model.encode.return_value = np.array([[1., 0.]], dtype="float32")
        retriever.index = Mock(ntotal=3)
        retriever.index.search.return_value = (np.array([[.99, .8, .7]]), np.array([[0, 1, 2]]))
        retriever.metadata = [
            dict(chunk_id=cid, control_id=cid, family=cid[:2], title=cid, text=cid, is_withdrawn=withdrawn)
            for cid, withdrawn in (("SA-12", True), ("RA-3(1)", False), ("PM-30", False))
        ]
        self.assertEqual([d.control_id for d in retriever.retrieve("supply chain", k=2)], ["RA-3(1)", "PM-30"])
        self.assertEqual(retriever.retrieve("historical", k=1, include_withdrawn=True)[0].control_id, "SA-12")

    def test_duplicate_chunks_and_withdrawn_controls_do_not_inflate_similarity_signal(self):
        from dataclasses import replace
        pipeline = self.pipeline()
        one = pipeline._retriever.retrieve.return_value[0]
        docs = [one, replace(one, chunk_id="AC-2::second"),
                replace(one, control_id="SA-12", is_withdrawn=True)]
        self.assertEqual(pipeline._detect_retrieval_signal(docs), RetrievalSignal.MODERATE_SIMILARITY)

    def test_every_control_body_is_preserved_with_bounded_chunks(self):
        originals = {c.control_id: c for c in build_chunks()}
        grouped = defaultdict(list)
        for chunk in build_chunks(strategy="chars"):
            grouped[chunk.control_id].append(chunk)
        self.assertEqual(set(originals), set(grouped))
        for control_id, parts in grouped.items():
            original = originals[control_id]
            prefix = f"{control_id} {original.title}\n"
            end = len(prefix)
            for part in parts:
                self.assertLessEqual(len(part.text), 500)
                self.assertLessEqual(part.char_start, end)
                self.assertGreater(part.char_end, end)
                self.assertEqual(part.text, prefix + original.text[part.char_start:part.char_end])
                end = part.char_end
            self.assertEqual(end, len(original.text))

    def test_splitter_handles_no_spaces_and_rejects_nonprogressing_overlap(self):
        text = "한글" * 300
        spans = list(split_text_spans(text, 50, 10))
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        for previous, current in zip(spans, spans[1:]):
            self.assertEqual(previous[1] - current[0], 10)
        with self.assertRaises(ValueError):
            list(split_text_spans(text, 50, 50))

    def test_question_plan_is_repeatable_and_varied(self):
        chunks = build_chunks()
        plan = make_question_plan(chunks, 15, 42)
        self.assertEqual(plan, make_question_plan(chunks, 15, 42))
        self.assertEqual(len({p["question_type"] for p in plan}), 5)
        self.assertEqual(len({p["source_excerpts"][0]["family"] for p in plan}), 15)
        for item in plan:
            for source in item["source_excerpts"]:
                self.assertNotIn("[Withdrawn:", source["text"])

    def test_generation_retries_invalid_sources_and_saves_unreviewed_cases(self):
        item = {"question_type": "direct", "source_excerpts": [
            {"control_id": "AC-2", "text": "AC-2 Account management"}
        ]}
        draft = {"query": "How should accounts be managed?", "source_control_ids": ["FAKE-1"],
                 "reference_answer_draft": "Manage accounts. [AC-2]", "rationale": "Account requirements."}
        bad = json.dumps(draft)
        draft["source_control_ids"] = ["AC-2"]
        client = Mock()
        client.chat.side_effect = [
            {"message": {"content": bad}},
            {"message": {"content": json.dumps(draft)}},
        ]
        checkpoint = Mock()
        cases = generate_questions([item], client, on_progress=checkpoint)
        self.assertEqual(client.chat.call_count, 2)
        self.assertIsNone(cases[0]["expected"])
        self.assertEqual(cases[0]["review_status"], "unreviewed")
        checkpoint.assert_called_once_with(cases)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps({"complete": True, "cases": cases}))
            self.assertEqual(load_test_cases(path), cases)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_draft(json.dumps(draft), item, {draft["query"].casefold()})

    def test_multi_control_draft_must_reference_both_sources(self):
        item = {"question_type": "multi_control", "source_excerpts": [
            {"control_id": "AC-2"}, {"control_id": "AC-3"}
        ]}
        draft = {"query": "How do the controls interact?", "source_control_ids": ["AC-2"],
                 "reference_answer_draft": "Draft", "rationale": "Draft"}
        with self.assertRaisesRegex(ValueError, "both"):
            validate_draft(json.dumps(draft), item, set())


if __name__ == "__main__":
    unittest.main()
