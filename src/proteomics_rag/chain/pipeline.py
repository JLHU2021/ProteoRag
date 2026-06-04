"""RAG chain — LangChain orchestration: retriever → prompt → LLM → answer.

Assembles the full pipeline:
1. Route query (SQL vs RAG)
2. Retrieve relevant chunks (hybrid: dense + BM25 + reranker)
3. Build prompt with retrieved context
4. Call LLM (Gemini/Claude)
5. Return answer with citations
"""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from proteomics_rag.config import GOOGLE_API_KEY, ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, LLM_PROVIDER, LLM_MODEL
from proteomics_rag.retrieval.hybrid import HybridRetriever
from proteomics_rag.retrieval.router import SQLRouter, is_sql_query

logger = logging.getLogger(__name__)

# System prompt for the proteomics domain expert
SYSTEM_PROMPT = """\
You are ProteoRAG, a domain expert in proteomics and post-translational modifications (PTMs), \
particularly S-nitrosylation, arginine methylation, and cysteine oxidation.

Answer the user's question based ONLY on the provided context. \
If the context does not contain enough information to answer, say so explicitly. \
Always cite your sources using the format [source: filename, chunk_id].

Context:
{context}

Question: {question}
"""


def _build_llm(provider: str = LLM_PROVIDER, model: str = LLM_MODEL):
    """Build the LLM component based on configured provider."""
    if provider == "dashscope":
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY not set in .env")
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                api_key=DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.1,
            )
        except ImportError:
            raise ImportError(
                "Install langchain-openai: pip install langchain-openai"
            )
    elif provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.1,
            )
        except ImportError:
            raise ImportError(
                "Install langchain-google-genai: pip install langchain-google-genai"
            )
    elif provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model="claude-sonnet-4-20250514",
                anthropic_api_key=ANTHROPIC_API_KEY,
                temperature=0.1,
            )
        except ImportError:
            raise ImportError(
                "Install langchain-anthropic: pip install langchain-anthropic"
            )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'gemini' or 'anthropic'.")


class ProteoRAGChain:
    """Main RAG pipeline for proteomics Q&A."""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.router = SQLRouter()
        self._llm = None
        self._chain = None
        self._loaded = False

    def load(self):
        """Load all components (embedding models, indexes, LLM)."""
        if self._loaded:
            return
        logger.info("Loading ProteoRAG chain...")
        self.retriever.load()
        self._llm = _build_llm(model=LLM_MODEL)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        self._chain = prompt | self._llm | StrOutputParser()
        self._loaded = True
        logger.info("ProteoRAG chain loaded successfully")

    def query(self, question: str) -> dict[str, Any]:
        """Execute a query through the full pipeline.

        Returns:
            {
                "answer": str,
                "route": "rag" | "sql",
                "sources": list[dict],
                "context_used": str,
            }
        """
        if not self._loaded:
            self.load()

        # Step 1: Route
        sql_result = self.router.execute(question)
        if sql_result["route"] == "sql":
            return {
                "answer": sql_result["answer"],
                "route": "sql",
                "sources": [],
                "context_used": "",
            }

        # Step 2: Retrieve
        chunks = self.retriever.search(question)
        if not chunks:
            return {
                "answer": "No relevant documents found for your question.",
                "route": "rag",
                "sources": [],
                "context_used": "",
            }

        # Step 3: Build context
        context_parts = []
        sources = []
        for chunk in chunks:
            context_parts.append(
                f"[source: {chunk['source']}, id: {chunk['chunk_id']}]\n{chunk['text']}"
            )
            sources.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "score": chunk.get("score", 0),
            })

        context = "\n\n---\n\n".join(context_parts)

        # Step 4: Generate
        answer = self._chain.invoke({
            "context": context,
            "question": question,
        })

        return {
            "answer": answer,
            "route": "rag",
            "sources": sources,
            "context_used": context[:500] + "...",  # Truncate for display
        }


if __name__ == "__main__":
    chain = ProteoRAGChain()
    chain.load()
    result = chain.query("What is S-nitrosylation and how does it affect protein function?")
    print(f"\nRoute: {result['route']}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Sources: {len(result['sources'])}")
