"""IntelliBot - Streamlit app.

Day 1: document loading.  Day 2: chunking + embeddings.
Day 3: persistent ChromaDB index + similarity search.
Day 4: grounded LLM answers with source attribution.
Day 5: conversation memory for follow-up questions, per-session history, Clear chat.
Day 6: polished chat interface (chat bubbles, tabs, loading and error states, mobile layout).
Run locally:   streamlit run app.py

Layout:  tabs  💬 Chat | 📄 Documents | 🔎 Search   +  a status sidebar.
All retrieval / answering / memory logic lives in core/; this file only draws the screen.
"""

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import List

# Optional: some hosts ship an old system SQLite that ChromaDB cannot use. If the
# 'pysqlite3-binary' package is installed (see README, Troubleshooting), use it.
try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

from core.answering import ANSWERED, NOT_IN_DOCUMENTS, AnswerEngine
from core.chunking import ChunkingConfig, chunk_document
from core.config import SAMPLE_DOCS_DIR, load_settings
from core.conversation import ChatReply, ConversationEngine
from core.document_loader import LoadResult, load_document, load_path
from core.embeddings import (
    EmbeddingCache,
    EmbeddingError,
    SentenceTransformerEmbedder,
    embed_chunks,
    sentence_transformers_installed,
)
from core.indexing import index_chunks
from core.llm import LLMUnavailableError, create_llm_client
from core.memory import NEEDS_CLARIFICATION, ChatSession, Turn
from core.retrieval import (
    MAX_QUERY_CHARS,
    STATUS_EMPTY_QUERY,
    STATUS_NO_DOCUMENTS,
    STATUS_NO_RELEVANT,
    STATUS_OK,
    STATUS_QUERY_TOO_LONG,
    Retriever,
)
from core.ui_helpers import (
    CSS,
    EXAMPLE_QUESTIONS,
    Problem,
    describe_problem,
    history_limit_text,
    readiness,
    readiness_message,
    safe_details,
    setup_steps,
    unique_sources,
    unknown_problem,
)
from core.vector_store import ChromaVectorStore, VectorStoreError

st.set_page_config(
    page_title="IntelliBot", page_icon="🤖", layout="centered", initial_sidebar_state="auto"
)

USER_AVATAR, BOT_AVATAR = "🧑", "🤖"

SECRET_NAMES = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_MODEL",
    "EMBEDDING_MODEL",
    "MAX_UPLOAD_MB",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBEDDING_BATCH_SIZE",
    "TOP_K",
    "MIN_SIMILARITY",
    "VECTOR_STORE_DIR",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT_SECONDS",
    "HUGGINGFACE_BASE_URL",
    "CHAT_HISTORY_TURNS",
)


def apply_streamlit_secrets() -> None:
    """On Streamlit Cloud, copy Secrets into environment variables."""
    try:
        for name in SECRET_NAMES:
            if name in st.secrets and name not in os.environ:
                os.environ[name] = str(st.secrets[name])
    except Exception:
        pass  # no secrets file (normal when running locally)



def show_result(result: LoadResult) -> None:
    if not result.ok:
        st.error(f"❌ **{result.filename}** - {result.error}")
        return
    doc = result.document
    pages = "page" if doc.page_count == 1 else "pages"
    with st.expander(
        f"✅ {doc.name} - {doc.page_count} {pages}, {doc.char_count:,} characters"
    ):
        for warning in doc.warnings:
            st.warning(warning)
        st.caption(f"Type: {doc.file_type.upper()} · Read with: {doc.parser}")
        preview = doc.text[:700] + ("..." if len(doc.text) > 700 else "")
        st.text(preview)



# ---------------------------------------------------------------- drawing helpers
def mask_text(text: str) -> str:
    """Hide pasted/configured secrets in any text before it is drawn."""
    return state["chat"].mask(text)


def show_problem(problem: Problem) -> None:
    """A friendly message with a clear next step (never a stack trace or a secret)."""
    box = {"error": st.error, "warning": st.warning, "info": st.info}[problem.kind]
    box(f"**{problem.title}**\n\n{mask_text(problem.message)}")
    st.caption(f"👉 {problem.next_step}")


def show_source_card(source, full_text: bool) -> None:
    """One passage: real file name, page and text. A fresh source (SourceReference) also has
    chunk number and similarity; an earlier turn's source (SourceNote) keeps file, page, excerpt."""
    with st.container(border=True):
        st.markdown(f"**[{source.label}] {source.source}**")
        if hasattr(source, "chunk_index"):
            st.caption(
                f"{source.pages} · chunk {source.chunk_index + 1} of {source.total_chunks} · "
                f"similarity {source.score:.2f}"
            )
            body = source.text if full_text else source.excerpt
        else:
            st.caption(source.pages)
            body = source.excerpt
        st.text(mask_text(body))


def show_sources(cited, others=()) -> None:
    """Compact 'Sources' line + an expander with the passages (does not flood the chat)."""
    if cited:
        st.caption("📎 Sources: " + "; ".join(unique_sources(cited)))
    total = len(cited) + len(others)
    if not total:
        return
    with st.expander(f"View sources and excerpts ({total})"):
        for source in cited:
            show_source_card(source, full_text=True)
        if others:
            st.markdown("**Also checked, not used as support:**")
            for source in others:
                show_source_card(source, full_text=False)


def show_assistant_reply(reply: ChatReply) -> None:
    """The newest answer (or problem) inside the assistant bubble, with full detail."""
    answer = reply.answer
    if reply.was_rewritten and reply.method != "clarify":
        st.caption(f"🔎 I understood your question as: {mask_text(reply.standalone_question)}")
    if answer.status == ANSWERED:
        st.write(mask_text(answer.text))
        for warning in answer.warnings:
            st.warning(mask_text(warning))
        show_sources(answer.cited_sources, [s for s in answer.sources if not s.cited])
    elif answer.status == NEEDS_CLARIFICATION:
        st.write(f"❓ {mask_text(answer.text)}")
        st.caption(f"👉 {describe_problem(answer.status, answer.text).next_step}")
    else:
        show_problem(describe_problem(answer.status, answer.text) or unknown_problem())
        for warning in answer.warnings:
            st.warning(mask_text(warning))
        if answer.sources:
            label = (
                "Passages that were checked (not used as support)"
                if answer.status == NOT_IN_DOCUMENTS
                else "Passages found in your documents (no answer was written)"
            )
            with st.expander(f"{label} ({len(answer.sources)})"):
                for source in answer.sources:
                    show_source_card(source, full_text=False)
    for note in reply.notes:
        st.caption(mask_text(note))
    with st.expander("🔧 Technical details"):
        for key, value in safe_details(answer, reply.standalone_question, reply.method).items():
            st.text(f"{key}: {value}")


def show_turn(turn: Turn) -> None:
    """An earlier exchange of this chat (question + answer + sources), shown compactly."""
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(mask_text(turn.question))
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        if turn.status == ANSWERED:
            st.write(mask_text(turn.answer_text))
            for warning in turn.warnings:
                st.warning(mask_text(warning))
            if turn.sources:
                show_sources(turn.sources)
        elif turn.status == NEEDS_CLARIFICATION:
            st.write(f"❓ {mask_text(turn.answer_text)}")
        else:
            show_problem(describe_problem(turn.status, turn.answer_text) or unknown_problem())


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str, batch_size: int) -> SentenceTransformerEmbedder:
    """One embedder (and so one loaded model) shared by the app. Loads lazily."""
    return SentenceTransformerEmbedder(model_name, batch_size)


@st.cache_resource(show_spinner=False)
def get_vector_store(persist_dir: str, embedding_model: str) -> ChromaVectorStore:
    """One ChromaDB connection shared by the whole app (one process = one client)."""
    return ChromaVectorStore(Path(persist_dir), embedding_model)


apply_streamlit_secrets()
settings = load_settings()

# Connect to the persistent ChromaDB index (if ChromaDB is installed and works).
store = None
store_error = None
try:
    store = get_vector_store(str(settings.vector_store_path), settings.embedding_model)
except VectorStoreError as error:
    store_error = str(error)


def stored_chunk_count() -> int:
    try:
        return store.count() if store is not None else 0
    except VectorStoreError:
        return 0

# Per-browser-session storage (one visitor never sees another visitor's files).
state = st.session_state
state.setdefault("load_cache", {})  # (name, sha256) -> LoadResult
state.setdefault("use_samples", False)
state.setdefault("chunk_cache", {})  # (doc hash, size, overlap) -> list of Chunk
state.setdefault("embedding_cache", EmbeddingCache())  # reuse vectors, per session
state.setdefault("embedding_summary", None)
state.setdefault("last_search", None)
state.setdefault("last_reply", None)  # newest exchange, drawn in full
# This browser session's own conversation. Kept only in memory (st.session_state) and never
# shared with other sessions or written to disk. Known secrets are masked before storing.
state.setdefault(
    "chat",
    ChatSession(
        max_turns=settings.chat_history_turns,
        redact_values=[settings.openai_api_key, settings.huggingface_api_token],
    ),
)


def clear_chat() -> None:
    """Forget THIS session's conversation only (not the index, documents or other sessions)."""
    state["chat"].clear()
    state["last_reply"] = None
    state["notice"] = "🧹 Chat cleared. Your documents and the saved index were not changed."


def ask_question(prompt: str) -> ChatReply:
    """Run one question through the conversation engine (follow-ups, retrieval, grounded answer)."""
    llm, llm_reason = None, ""
    try:
        llm = create_llm_client(settings)
    except LLMUnavailableError as error:
        llm_reason = str(error)
    embedder = get_embedder(settings.embedding_model, settings.embedding_batch_size)
    engine = ConversationEngine(
        AnswerEngine(
            Retriever(store, embedder, top_k=settings.top_k, min_score=settings.min_similarity),
            llm,
            llm_unavailable_reason=llm_reason,
        )
    )
    return engine.ask(state["chat"], prompt, doc_hashes=session_hashes)


st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)  # the ONE static stylesheet

# ---------------------------------------------------------------- header + tabs
st.title("🤖 IntelliBot")
st.caption("Ask questions about your own documents. Answers come only from them, with sources.")
status_area = st.container()  # filled at the end, when the setup state is known
tab_chat, tab_docs, tab_search = st.tabs(["💬 Chat", "📄 Documents", "🔎 Search"])

# ================================================================ 📄 Documents tab
with tab_docs:
    st.subheader("1. Add your documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help=f"Up to {settings.max_upload_mb} MB per file.",
    )

    col_a, col_b = st.columns(2)
    if col_a.button("Use sample documents", use_container_width=True):
        state["use_samples"] = True
    if state["use_samples"] and col_b.button("Remove samples", use_container_width=True):
        state["use_samples"] = False

    results: List[LoadResult] = []

    for uploaded in uploaded_files or []:
        data = uploaded.getvalue()
        cache_key = (uploaded.name, hashlib.sha256(data).hexdigest())
        if cache_key not in state["load_cache"]:
            with st.spinner(f"Reading {uploaded.name}..."):
                state["load_cache"][cache_key] = load_document(
                    uploaded.name, data, max_bytes=settings.max_upload_bytes
                )
        results.append(state["load_cache"][cache_key])

    if state["use_samples"]:
        for path in sorted(SAMPLE_DOCS_DIR.glob("*")):
            if path.suffix.lower() not in settings.supported_extensions:
                continue
            cache_key = ("sample", path.name)
            if cache_key not in state["load_cache"]:
                state["load_cache"][cache_key] = load_path(
                    path, max_bytes=settings.max_upload_bytes
                )
            results.append(state["load_cache"][cache_key])

    # Keep only successfully loaded documents, skipping exact duplicates.
    documents = []
    seen_hashes = set()
    for result in results:
        if result.ok and result.document.content_hash not in seen_hashes:
            seen_hashes.add(result.document.content_hash)
            documents.append(result.document)
    state["documents"] = documents  # used by the indexing step on Day 2

    if results:
        failed = sum(1 for r in results if not r.ok)
        st.write(
            f"**{len(documents)}** document(s) ready"
            + (f" · **{failed}** could not be read" if failed else "")
        )
        for result in results:
            show_result(result)
    else:
        st.info("No documents yet. Upload a file above, or tap **Use sample documents**.")

    # ---------------------------------------------------------------- Day 2: chunks
    st.subheader("2. Split into chunks")
    chunks = []
    chunks_by_doc = {}  # file hash -> its chunks (used to tell whether a document is indexed)
    if documents:
        config = ChunkingConfig(settings.chunk_size, settings.chunk_overlap)
        st.caption(
            f"Each document is cut into pieces of up to {config.chunk_size} characters, "
            f"overlapping by {config.chunk_overlap}."
        )
        for doc in documents:
            cache_key = (doc.content_hash, config.chunk_size, config.chunk_overlap)
            if cache_key not in state["chunk_cache"]:
                state["chunk_cache"][cache_key] = chunk_document(doc, config)
            doc_chunks = state["chunk_cache"][cache_key]
            chunks_by_doc[doc.content_hash] = doc_chunks
            chunks.extend(doc_chunks)
            with st.expander(f"✂️ {doc.name} - {len(doc_chunks)} chunks"):
                for chunk in doc_chunks[:5]:
                    pages = (
                        f"page {chunk.page_start}"
                        if chunk.page_start == chunk.page_end
                        else f"pages {chunk.page_start}-{chunk.page_end}"
                    )
                    st.caption(
                        f"Chunk {chunk.chunk_index + 1} of {chunk.total_chunks} · {pages} · "
                        f"{len(chunk.text)} characters"
                    )
                    st.text(chunk.text)
                if len(doc_chunks) > 5:
                    st.caption(f"...and {len(doc_chunks) - 5} more chunks.")
        st.write(f"**{len(chunks)}** chunks in total")
    else:
        st.info("Add documents first to see how they are split into chunks.")
    state["chunks"] = chunks  # used by the vector store on Day 3

    # ---------------------------------------------------------------- Day 2 + 3: embeddings and index
    st.subheader("3. Create embeddings and index")
    session_hashes = [doc.content_hash for doc in documents]
    can_embed = sentence_transformers_installed()
    if not chunks:
        st.info("Embeddings are created from chunks. Add documents first.")
    else:
        st.caption(
            "Embeddings are numbers that capture the meaning of each chunk. They run on this "
            "machine (no API key). The first time, a model of about 90 MB is downloaded. "
            "The chunks and embeddings are then saved in a ChromaDB index, so they are "
            "not recreated next time."
        )
        if not can_embed:
            st.warning(
                "Embeddings are not available here because the 'sentence-transformers' package "
                "is not installed. Install the requirements and restart the app."
            )
        if store is None:
            st.warning(
                "Indexing and search are not available because ChromaDB could not be started. "
                + (store_error or "")
            )
        button_label = "Create embeddings and index" if store is not None else "Create embeddings"
        if st.button(button_label, disabled=not can_embed, use_container_width=True):
            try:
                embedder = get_embedder(settings.embedding_model, settings.embedding_batch_size)
                started = time.time()
                with st.spinner("Working (the first run may take a while)..."):
                    if store is not None:
                        report = index_chunks(chunks, store, embedder, state["embedding_cache"])
                        summary = {
                            "kind": "index",
                            "documents_indexed": report.documents_indexed,
                            "documents_unchanged": report.documents_unchanged,
                            "chunks_written": report.chunks_written,
                            "new": report.newly_embedded,
                            "reused": report.reused_vectors,
                            "dimension": report.dimension,
                            "store_total": report.store_total_chunks,
                        }
                    else:  # Day 2 behaviour: embeddings only, kept in memory
                        result = embed_chunks(chunks, embedder, state["embedding_cache"])
                        state["embedded_chunks"] = result.items
                        summary = {
                            "kind": "embed",
                            "total": len(result.items),
                            "new": result.newly_embedded,
                            "reused": result.reused,
                            "dimension": result.dimension,
                        }
                summary["ids"] = tuple(c.chunk_id for c in chunks)
                summary["seconds"] = time.time() - started
                state["embedding_summary"] = summary
                state["last_search"] = None
            except (EmbeddingError, VectorStoreError) as error:
                state["embedding_summary"] = None
                st.error(f"❌ {error}")
        summary = state["embedding_summary"]
        if summary and summary["ids"] == tuple(c.chunk_id for c in chunks):
            if summary["kind"] == "index":
                st.success(
                    f"✅ {summary['documents_indexed']} document(s) indexed, "
                    f"{summary['documents_unchanged']} already up to date · "
                    f"{summary['chunks_written']} chunks saved "
                    f"({summary['new']} newly embedded, {summary['reused']} reused) · "
                    f"vector size {summary['dimension'] or '-'} · {summary['seconds']:.1f} s · "
                    f"the index now holds {summary['store_total']} chunks"
                )
                st.info("👉 Next: open the 💬 Chat tab at the top and ask a question about your documents.")
            else:
                st.success(
                    f"✅ {summary['total']} chunks embedded "
                    f"({summary['new']} new, {summary['reused']} reused) · "
                    f"vector size {summary['dimension']} · {summary['seconds']:.1f} s "
                    "(not saved: no ChromaDB)"
                )
        elif summary:
            st.info("Your documents changed. Tap the button again.")

    if store is not None and chunks:
        with st.expander("Manage the saved index"):
            if st.button("Remove my documents from the index", use_container_width=True):
                try:
                    removed = sum(store.delete_document(h) for h in session_hashes)
                    state["embedding_summary"] = None
                    state["last_search"] = None
                    st.success(f"Removed {removed} saved chunks. You can index again any time.")
                except VectorStoreError as error:
                    st.error(f"❌ {error}")
            st.caption(
                "Advanced: clearing deletes EVERY saved chunk, including other people's if "
                "this app is shared. Documents can always be indexed again."
            )
            confirm_clear = st.checkbox("I understand, clear the whole index")
            if st.button("Clear the whole index", disabled=not confirm_clear, use_container_width=True):
                try:
                    store.clear()
                    state["embedding_summary"] = None
                    state["last_search"] = None
                    st.success("The index is empty.")
                except VectorStoreError as error:
                    st.error(f"❌ {error}")


# Which of this session's documents are already indexed? (drives the setup hints)
n_indexed = 0
if store is not None and documents:
    try:
        n_indexed = sum(
            1 for d in documents if store.is_document_current(chunks_by_doc.get(d.content_hash, []))
        )
    except VectorStoreError:
        n_indexed = 0

# ================================================================ 🔎 Search tab
with tab_search:
    # ---------------------------------------------------------------- Day 3: search
    st.subheader("4. Search your documents")
    if store is None:
        st.info("Search needs ChromaDB (see the 📄 Documents tab).")
    elif not can_embed:
        st.info("Search needs the embedding model (see the 📄 Documents tab).")
    elif not documents:
        st.info("Add and index documents first, then search them here.")
    else:
        st.caption(
            "Finds the parts of your documents that are most similar in meaning to your "
            "text. Only the documents you added in this session are searched."
        )
        query = st.text_input("Type a question or keywords", max_chars=MAX_QUERY_CHARS)
        col_k, col_sim = st.columns(2)
        top_k = col_k.slider("Results (top-K)", 1, max(10, settings.top_k), settings.top_k)
        min_similarity = col_sim.slider(
            "Minimum similarity", 0.0, 1.0, float(settings.min_similarity), 0.05
        )
        if st.button("Search", use_container_width=True):
            try:
                embedder = get_embedder(settings.embedding_model, settings.embedding_batch_size)
                with st.spinner("Searching..."):
                    response = Retriever(store, embedder).retrieve(
                        query, top_k=top_k, min_score=min_similarity, doc_hashes=session_hashes
                    )
                state["last_search"] = {"response": response, "hashes": tuple(session_hashes)}
            except (EmbeddingError, VectorStoreError) as error:
                state["last_search"] = None
                st.error(f"❌ {error}")

        last = state["last_search"]
        if last and last["hashes"] == tuple(session_hashes):
            response = last["response"]
            if response.status in (STATUS_EMPTY_QUERY, STATUS_QUERY_TOO_LONG):
                st.warning(response.message)
            elif response.status == STATUS_NO_DOCUMENTS:
                st.info(response.message)
            elif response.status == STATUS_NO_RELEVANT:
                st.warning(
                    f"{response.message} (best similarity {response.best_score:.2f}, "
                    f"minimum {response.min_score:.2f}). Try other words or lower the minimum."
                )
            elif response.status == STATUS_OK:
                st.success(
                    f"{len(response.results)} matching part(s) found in {response.elapsed_seconds:.2f} s"
                )
                for result in response.results:
                    pages = (
                        f"page {result.page_start}"
                        if result.page_start == result.page_end
                        else f"pages {result.page_start}-{result.page_end}"
                    )
                    with st.container(border=True):
                        st.markdown(f"**#{result.rank} · {result.source}**")
                        st.caption(
                            f"similarity {result.score:.2f} · {pages} · "
                            f"chunk {result.chunk_index + 1} of {result.total_chunks}"
                        )
                        st.text(result.text)
        elif last:
            st.info("Your documents changed. Search again.")


# ================================================================ 💬 Chat tab
with tab_chat:
    chat = state["chat"]
    chat_ready = store is not None and can_embed
    setup_state = readiness(len(documents), n_indexed, store is not None, can_embed)
    banner = readiness_message(setup_state)
    if banner:
        {"info": st.info, "warning": st.warning}[banner[0]](banner[1])
    elif not settings.has_llm_credentials:
        st.warning(
            "🔑 No API key found, so written answers are not available. "
            + settings.llm_status_message()
            + " Searching (🔎 Search tab) still works."
        )

    notice = state.pop("notice", None)
    if notice:
        st.success(notice)

    st.caption(history_limit_text(settings.chat_history_turns, len(chat)))
    st.button(
        "🧹 Clear chat",
        key="clear_chat_main",
        on_click=clear_chat,
        disabled=chat.is_empty and not state["last_reply"],
        use_container_width=True,
        help="Forget this conversation. Your documents and the saved index are not changed.",
    )

    messages_area = st.container()  # filled after a new question has been processed
    prompt = st.chat_input(
        "Ask a question about your documents…" if chat_ready else "Set up documents first (see above)",
        max_chars=MAX_QUERY_CHARS,
        disabled=not chat_ready,
    )

    last = state["last_reply"]
    current = last if last and last["hashes"] == tuple(session_hashes) else None
    with messages_area:
        if prompt:  # a NEW question (st.chat_input returns it once, so reruns cannot repeat it)
            for turn in chat.turns:
                show_turn(turn)
            with st.chat_message("user", avatar=USER_AVATAR):
                st.write(chat.mask(prompt.strip()))  # masked BEFORE it is shown
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                new_reply = None
                try:
                    with st.spinner("Searching your documents and writing the answer…"):
                        new_reply = ask_question(prompt)
                except VectorStoreError as error:
                    show_problem(
                        Problem(
                            "error",
                            "The search index could not be opened",
                            str(error),
                            "Open 📄 Documents → “Manage the saved index” and index again.",
                        )
                    )
                if new_reply is not None:
                    state["last_reply"] = {
                        "reply": new_reply,
                        "recorded": new_reply.recorded,
                        "hashes": tuple(session_hashes),
                    }
                    show_assistant_reply(new_reply)
                else:
                    state["last_reply"] = None
        else:
            earlier = chat.turns
            if current and current["recorded"] and earlier:
                earlier = earlier[:-1]  # the newest exchange is drawn in full below
            for turn in earlier:
                show_turn(turn)
            if current:
                shown_reply = current["reply"]
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.write(mask_text(shown_reply.answer.question))
                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    show_assistant_reply(shown_reply)
            elif last:
                st.caption("Your documents changed since your last question; ask again.")
            if not chat.turns and not current and chat_ready and documents:
                if state["use_samples"]:
                    st.caption("💡 Try asking, for example:")
                    for example in EXAMPLE_QUESTIONS:
                        st.caption(f"• {example}")
                else:
                    st.caption("💡 Type a question below to get started.")

# ================================================================ status strip + sidebar
with status_area:
    steps = setup_steps(
        len(documents), n_indexed, store is not None, can_embed, settings.has_llm_credentials
    )
    st.markdown("  ·  ".join(f"{step.mark} {step.label}" for step in steps))

with st.sidebar:
    st.markdown("### 🤖 IntelliBot")
    st.caption("Answers only from your own documents.")
    st.markdown("**Setup**")
    for step in steps:
        st.markdown(f"{step.mark} {step.label}")
        if not step.done:
            st.caption(step.hint)

    st.markdown("**Status**")
    if settings.has_llm_credentials:
        st.success("API key found ✔")
    else:
        st.warning("No API key yet")
        st.caption(settings.llm_status_message())
        st.caption("You can still upload, index and search documents. A key is needed only for written answers.")
    st.caption(f"Max file size: {settings.max_upload_mb} MB")
    st.caption("Supported files: PDF, TXT")
    st.caption(f"Chunks: {settings.chunk_size} characters, overlap {settings.chunk_overlap}")
    st.caption(f"Embedding model: {settings.embedding_model}")
    st.caption(f"Answer model: {settings.llm_provider} · {settings.active_model}")
    if store is not None:
        st.caption(f"Search index: ChromaDB · {stored_chunk_count()} chunks saved")
    else:
        st.caption("Search index: not available (ChromaDB)")

    st.markdown("**Chat**")
    st.caption(history_limit_text(settings.chat_history_turns, len(state["chat"])))
    st.button(
        "🧹 Clear chat",
        key="clear_chat_sidebar",
        on_click=clear_chat,
        disabled=state["chat"].is_empty and not state["last_reply"],
        use_container_width=True,
    )
    with st.expander("How to use IntelliBot"):
        st.markdown(
            "1. Open **📄 Documents**, add PDF/TXT files (or the samples).\n"
            "2. Tap **Create embeddings and index**.\n"
            "3. Go to **💬 Chat** and ask a question.\n"
            "4. Ask follow-ups like *“What about the second option?”*.\n"
            "5. Open **View sources** under an answer to check it.\n"
            "6. **Clear chat** forgets the conversation (documents stay)."
        )
