import time
from dataclasses import dataclass, field
from enum import Enum
import ollama
from retriever import NISTRetriever, RetrievedChunk

OLLAMA_MODEL = "llama3"
THRESHOLD_FULL = 0.55
THRESHOLD_PARTIAL = 0.30
MIN_DOCS_FULL = 2

class SupportLevel(str, Enum):
    FULLY_SUPPORTED = "FULLY_SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    NO_INFORMATION = "NO_INFORMATION"

@dataclass
class RAGResponse:
    query: str
    answer: str
    support_level: SupportLevel
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0

class RAGPipeline:
    def __init__(self, model: str = OLLAMA_MODEL):
        self._model = model
        self._retriever = NISTRetriever()

    def query(self, query_text: str) -> RAGResponse:
        docs = self._retriever.retrieve(query_text, k=5)
        top_score = docs[0].score if docs else 0.0
        
        support_level = self._detect_support(docs)
        
        if support_level == SupportLevel.NO_INFORMATION:
            return RAGResponse(
                query=query_text,
                answer="I'm sorry, but I couldn't find any relevant information in the NIST 800-53 catalog regarding your request.",
                support_level=support_level,
                retrieved=docs,
                top_score=top_score
            )
        
        context = self._build_context(docs)
        answer = self._generate(query_text, context)
        
        return RAGResponse(
            query=query_text,
            answer=answer,
            support_level=support_level,
            retrieved=docs,
            top_score=top_score
        )
    def _detect_support(self, docs: list[RetrievedChunk]) -> SupportLevel:
        if not docs or docs[0].score < THRESHOLD_PARTIAL:
            return SupportLevel.NO_INFORMATION
        high_count = sum(1 for d in docs if d.score >= THRESHOLD_FULL)
        if high_count >= MIN_DOCS_FULL:
            return SupportLevel.FULLY_SUPPORTED
        return SupportLevel.PARTIALLY_SUPPORTED

    def _build_context(self, docs: list[RetrievedChunk]) -> str:
        return "\n\n".join([f"[{d.control_id}] {d.title}: {d.text}" for d in docs])

    def _generate(self, query: str, context: str) -> str:
        system_message = (
            "You are a NIST 800-53 security compliance expert.\n"
            "Your goal is to provide accurate answers based ONLY on the provided context.\n"
            "STRICT RULES:\n"
            "1. EVERY factual claim MUST cite its source using the format [Control_ID] at the end of EVERY sentence. \n"
            "2. Example: 'The organization must define the types of events to be logged. [AU-2]' \n"
            "3. If you summarize multiple requirements, cite each one: 'It includes access control [AC-1] and audit logging [AU-2].'\n"
            "4. DO NOT provide any information that you cannot back up with a [Control_ID] from the context. \n"
            "5. If the context is insufficient, state 'Information not found in the provided catalog.'"
        )

        user_prompt = (
            f"Context from NIST 800-53 catalog:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a grounded answer with mandatory inline citations for each claim:"
        )

        response = ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0} 
        )
        return response["message"]["content"].strip()