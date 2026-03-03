"""Streamlit UI for the EDR (Engineering Document Review) Platform."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from src.agents import compile_graph
from src.ingestion import chunk_documents, create_index, load_pdf, get_vector_store
from src.models import ComplianceVerdict, Verdict, Severity
from src.tools import retrieve_documents as retrieve_documents_basic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_live_findings(
    placeholder,
    verdicts: list,
    topics_done: int,
    topic_count: int,
    current_topic: str,
) -> None:
    """Re-render live findings inside an st.empty() placeholder (replaces previous content).

    Uses only non-stateful elements (st.markdown, st.metric, st.progress) so that
    repeated replacement of the placeholder does not raise DuplicateWidgetID errors.
    """
    with placeholder.container():
        label = (
            f"Topic {topics_done}/{topic_count} complete — now processing: **{current_topic}**"
            if topic_count > 0
            else "Planning review topics..."
        )
        st.progress(
            value=(topics_done / topic_count) if topic_count else 0.0,
            text=label,
        )

        if verdicts:
            compliant = sum(
                1 for v in verdicts
                if isinstance(v, ComplianceVerdict) and v.verdict == Verdict.COMPLIANT
            )
            non_compliant = sum(
                1 for v in verdicts
                if isinstance(v, ComplianceVerdict) and v.verdict == Verdict.NON_COMPLIANT
            )
            partial = sum(
                1 for v in verdicts
                if isinstance(v, ComplianceVerdict) and v.verdict == Verdict.PARTIAL
            )
            not_addressed = sum(
                1 for v in verdicts
                if isinstance(v, ComplianceVerdict) and v.verdict == Verdict.NOT_ADDRESSED
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("✅ Compliant", compliant)
            c2.metric("❌ Non-Compliant", non_compliant)
            c3.metric("⚠️ Partial", partial)
            c4.metric("❓ Not Addressed", not_addressed)

            st.caption(
                f"{len(verdicts)} findings from {topics_done} of {topic_count} topics so far"
            )
            st.markdown("**Latest findings:**")

            color_map = {
                Verdict.COMPLIANT: "🟢",
                Verdict.NON_COMPLIANT: "🔴",
                Verdict.PARTIAL: "🟡",
                Verdict.NOT_ADDRESSED: "⚪",
            }
            for v in reversed(verdicts[-20:]):
                if not isinstance(v, ComplianceVerdict):
                    continue
                icon = color_map.get(v.verdict, "⚪")
                st.markdown(
                    f"{icon} **{v.req_id}** — {v.verdict.value} ({v.severity.value})  \n"
                    f"_{v.reasoning}_"
                )
        else:
            st.info("Waiting for first topic to complete...")



# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EDR — Engineering Document Review",
    page_icon="🏗️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = {}  # filename -> doc_type
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "review_result" not in st.session_state:
    st.session_state.review_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline_mode" not in st.session_state:
    st.session_state.pipeline_mode = "Basic"

# ---------------------------------------------------------------------------
# Sidebar — Document Upload & Indexing
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline Mode")
    pipeline_mode = st.radio(
        "Select pipeline",
        ["Basic", "Advanced"],
        index=["Basic", "Advanced"].index(st.session_state.pipeline_mode),
        help=(
            "**Basic:** PyPDF + flat chunks + dense search.\n\n"
            "**Advanced:** Azure Doc Intelligence + hierarchical chunks + hybrid BM25/dense search."
        ),
        key="pipeline_radio",
    )
    st.session_state.pipeline_mode = pipeline_mode

    st.divider()
    st.header("📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.text(f.name[:30])
            with col2:
                doc_type = st.selectbox(
                    "Type",
                    ["spec", "proposal"],
                    key=f"type_{f.name}",
                    label_visibility="collapsed",
                )
                st.session_state.indexed_docs[f.name] = doc_type

    if st.button("🔍 Process Documents", disabled=not uploaded_files, use_container_width=True):
        all_chunks = []
        progress = st.progress(0)

        if st.session_state.pipeline_mode == "Advanced":
            from src.ingestion_advanced import (
                load_pdf_with_layout,
                build_section_tree,
                chunk_hierarchical,
                create_index_advanced,
            )

            for i, f in enumerate(uploaded_files):
                doc_type = st.session_state.indexed_docs.get(f.name, "spec")

                tmp_path = Path(tempfile.gettempdir()) / f"edr_{f.name}"
                tmp_path.write_bytes(f.getvalue())

                with st.spinner(f"Analyzing layout of {f.name} with Azure Document Intelligence..."):
                    result = load_pdf_with_layout(tmp_path)
                    sections = build_section_tree(result)
                    parents, children = chunk_hierarchical(sections, doc_type=doc_type, source=f.name)
                    all_chunks.extend(parents + children)

                progress.progress((i + 1) / len(uploaded_files))

            st.text(f"Indexing {len(all_chunks)} chunks (hybrid BM25 + dense)...")
            create_index_advanced(all_chunks)
        else:
            for i, f in enumerate(uploaded_files):
                doc_type = st.session_state.indexed_docs.get(f.name, "spec")
                st.text(f"Loading {f.name}...")

                tmp_path = Path(tempfile.gettempdir()) / f"edr_{f.name}"
                tmp_path.write_bytes(f.getvalue())

                docs = load_pdf(tmp_path)
                chunks = chunk_documents(docs, doc_type=doc_type)
                all_chunks.extend(chunks)

                progress.progress((i + 1) / len(uploaded_files))

            st.text(f"Indexing {len(all_chunks)} chunks...")
            create_index(all_chunks)

        st.session_state.all_chunks = all_chunks
        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} documents")

    # Show indexed status
    if st.session_state.all_chunks:
        st.divider()
        st.metric("Indexed Chunks", len(st.session_state.all_chunks))
        st.caption(f"Pipeline: **{st.session_state.pipeline_mode}**")
        for name, dtype in st.session_state.indexed_docs.items():
            st.text(f"  {dtype}: {name[:35]}")

        if st.button("🗑️ Clear All & Start Fresh", use_container_width=True, type="secondary"):
            # Reset module-level vector stores / singletons
            import src.ingestion as _ing_basic
            _ing_basic._vector_store = None
            _ing_basic._qdrant_client = None

            try:
                import src.ingestion_advanced as _ing_adv
                _ing_adv._vector_store_advanced = None
                _ing_adv._section_outlines.clear()
            except Exception:
                pass

            # Reset all session state
            st.session_state.all_chunks = []
            st.session_state.indexed_docs = {}
            st.session_state.review_result = None
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main area — Tabs
# ---------------------------------------------------------------------------
st.title("🏗️ EDR — Engineering Document Review")

tab_review, tab_chat = st.tabs(["📋 Compliance Review", "💬 Chat with Documents"])

# ---------------------------------------------------------------------------
# Tab 1: Compliance Review
# ---------------------------------------------------------------------------
with tab_review:
    if not st.session_state.all_chunks:
        st.info("Upload and index documents using the sidebar to begin.")
    else:
        review_request = st.text_area(
            "Review Request",
            value="Review the proposal against the specification requirements for compliance.",
            height=80,
        )

        if st.button("🚀 Run Compliance Review", use_container_width=True):
            # Local accumulators — maintained across the streaming loop
            accumulated_verdicts: list = []
            accumulated_summaries: list = []
            topics_list: list[str] = []
            topics_done = 0
            current_topic = "..."
            final_report = ""

            # live_placeholder: re-rendered in-place after each topic completes
            # step_log: append-only status box showing node-level progress
            live_placeholder = st.empty()
            step_log = st.status("Running compliance review...", expanded=True)

            try:
                if st.session_state.pipeline_mode == "Advanced":
                    from src.tools_advanced import retrieve_documents as adv_retrieve
                    graph = compile_graph(retrieval_tool=adv_retrieve)
                else:
                    graph = compile_graph()
                initial_state = {
                    "document_types": st.session_state.indexed_docs,
                    "review_request": review_request,
                    "topics": [],
                    "current_topic_index": 0,
                    "all_verdicts": [],
                    "topic_summaries": [],
                    "pending_topic_verdicts": [],
                    "supervisor_notes": [],
                    "final_report": "",
                    "messages": [],
                }

                for chunk in graph.stream(initial_state, stream_mode="updates"):
                    # Each chunk is {node_name: node_return_dict}
                    node_name, node_output = next(iter(chunk.items()))

                    if node_name == "plan_topics":
                        topics_list = node_output.get("topics", [])
                        current_topic = topics_list[0] if topics_list else "..."
                        step_log.write(
                            f"📋 Planned **{len(topics_list)} topics**: {', '.join(topics_list)}"
                        )
                        _render_live_findings(
                            live_placeholder, accumulated_verdicts,
                            topics_done, len(topics_list), current_topic,
                        )

                    elif node_name == "process_topic":
                        # operator.add reducer → chunk contains ONLY the new delta verdicts
                        new_verdicts = node_output.get("all_verdicts", [])
                        accumulated_verdicts.extend(new_verdicts)
                        accumulated_summaries.extend(node_output.get("topic_summaries", []))
                        topics_done += 1
                        step_log.write(
                            f"✅ Topic {topics_done}/{len(topics_list)} done: "
                            f"**{current_topic}** — {len(new_verdicts)} findings"
                        )
                        _render_live_findings(
                            live_placeholder, accumulated_verdicts,
                            topics_done, len(topics_list), current_topic,
                        )

                    elif node_name == "advance_topic":
                        next_idx = node_output.get("current_topic_index", 0)
                        if next_idx < len(topics_list):
                            current_topic = topics_list[next_idx]
                        _render_live_findings(
                            live_placeholder, accumulated_verdicts,
                            topics_done, len(topics_list), current_topic,
                        )

                    elif node_name == "supervisor_verify":
                        notes = node_output.get("supervisor_notes", [])
                        if notes:
                            # Strip the "[SupervisorVerify] Topic '...':" prefix for brevity
                            preview = notes[-1]
                            # Show first 200 chars of the supervisor's comment
                            colon_idx = preview.find(": ", preview.find("':") + 2)
                            short = preview[colon_idx + 2: colon_idx + 202] if colon_idx != -1 else preview[:200]
                            step_log.write(f"🔍 Supervisor verified **{current_topic}** — {short}...")

                    elif node_name == "generate_report":
                        final_report = node_output.get("final_report", "")
                        step_log.write("📄 Generating final compliance report...")

                    # "check_more_topics" and other routing nodes return empty dicts — ignored

                # ── Streaming complete ──────────────────────────────────────
                # Build result dict with same keys as graph.invoke() used to return,
                # so the existing static results section (below) works unchanged.
                st.session_state.review_result = {
                    "all_verdicts": accumulated_verdicts,
                    "topic_summaries": accumulated_summaries,
                    "topics": topics_list,
                    "final_report": final_report,
                }
                step_log.update(label="Review complete!", state="complete", expanded=False)
                # Clear live view BEFORE setting session_state to prevent overlap on rerun
                live_placeholder.empty()

            except Exception as e:
                live_placeholder.empty()
                step_log.update(label=f"Review failed: {e}", state="error", expanded=True)
                if accumulated_verdicts:
                    # Preserve partial results so the user isn't left with nothing
                    st.session_state.review_result = {
                        "all_verdicts": accumulated_verdicts,
                        "topic_summaries": accumulated_summaries,
                        "topics": topics_list,
                        "final_report": final_report,
                    }
                    st.warning(
                        f"Review stopped after {topics_done}/{len(topics_list)} topics. "
                        f"Partial results ({len(accumulated_verdicts)} findings) shown below."
                    )
                else:
                    st.error(f"Review failed: {e}")

        # Display results
        if st.session_state.review_result:
            result = st.session_state.review_result

            all_verdicts = result.get("all_verdicts", [])
            final_report = result.get("final_report", "")

            if not all_verdicts and not final_report:
                st.warning("No results available yet.")
            else:
                # Summary metrics
                st.subheader("📊 Summary")
                col1, col2, col3, col4 = st.columns(4)

                compliant = sum(1 for v in all_verdicts if isinstance(v, ComplianceVerdict) and v.verdict == Verdict.COMPLIANT)
                non_compliant = sum(1 for v in all_verdicts if isinstance(v, ComplianceVerdict) and v.verdict == Verdict.NON_COMPLIANT)
                partial = sum(1 for v in all_verdicts if isinstance(v, ComplianceVerdict) and v.verdict == Verdict.PARTIAL)
                not_addressed = sum(1 for v in all_verdicts if isinstance(v, ComplianceVerdict) and v.verdict == Verdict.NOT_ADDRESSED)

                col1.metric("✅ Compliant", compliant)
                col2.metric("❌ Non-Compliant", non_compliant)
                col3.metric("⚠️ Partial", partial)
                col4.metric("❓ Not Addressed", not_addressed)

                # Verdict details
                st.subheader("📋 Findings")
                severity_filter = st.multiselect(
                    "Filter by severity",
                    [s.value for s in Severity],
                    default=[s.value for s in Severity],
                )

                for v in all_verdicts:
                    if not isinstance(v, ComplianceVerdict):
                        continue
                    if v.severity.value not in severity_filter:
                        continue

                    # Color by verdict
                    color_map = {
                        Verdict.COMPLIANT: "🟢",
                        Verdict.NON_COMPLIANT: "🔴",
                        Verdict.PARTIAL: "🟡",
                        Verdict.NOT_ADDRESSED: "⚪",
                    }
                    icon = color_map.get(v.verdict, "⚪")

                    with st.expander(f"{icon} {v.req_id} — {v.verdict.value} ({v.severity.value})"):
                        st.markdown(f"**Requirement:** {v.requirement_text}")
                        st.markdown(f"**Proposal Claim:** {v.proposal_claim}")
                        st.markdown(f"**Reasoning:** {v.reasoning}")
                        st.markdown(f"**Confidence:** {v.confidence:.0%}")

                # Report download
                if final_report:
                    st.subheader("📄 Full Report")
                    st.markdown(final_report)

                    st.download_button(
                        "📥 Download Report (Markdown)",
                        data=final_report,
                        file_name="compliance_report.md",
                        mime="text/markdown",
                    )

                # JSON export
                if all_verdicts:
                    verdicts_json = json.dumps(
                        [v.model_dump() for v in all_verdicts if isinstance(v, ComplianceVerdict)],
                        indent=2,
                    )
                    st.download_button(
                        "📥 Download Verdicts (JSON)",
                        data=verdicts_json,
                        file_name="verdicts.json",
                        mime="application/json",
                    )

# ---------------------------------------------------------------------------
# Tab 2: Chat with Documents
# ---------------------------------------------------------------------------
with tab_chat:
    if not st.session_state.all_chunks:
        st.info("Upload and index documents to enable document chat.")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if user_query := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Retrieve and answer
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    # Use retrieval tool based on pipeline mode
                    if st.session_state.pipeline_mode == "Advanced":
                        from src.tools_advanced import retrieve_documents as adv_retrieve
                        chat_tool = adv_retrieve
                    else:
                        chat_tool = retrieve_documents_basic
                    results = chat_tool.invoke({
                        "query": user_query,
                        "doc_type": "all",
                        "top_k": 5,
                    })

                    # Use LLM to generate answer (reuse configurable LLM)
                    from src.agents import _get_llm
                    from langchain_core.messages import HumanMessage, SystemMessage

                    llm = _get_llm()

                    response = llm.invoke([
                        SystemMessage(content=(
                            "You are an engineering document assistant. "
                            "Answer questions based on the retrieved document chunks. "
                            "Cite page numbers when possible. "
                            "If the answer isn't in the documents, say so."
                        )),
                        HumanMessage(content=(
                            f"Question: {user_query}\n\n"
                            f"Retrieved Documents:\n{results}"
                        )),
                    ])

                    answer = response.content
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
