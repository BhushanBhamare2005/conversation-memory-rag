from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import SAMPLE_DATA_PATH
from app.core.pipeline import MemoryPipeline
from app.chatbot.engine import ConversationalMemoryChatbot
from app.analysis.statistics import build_evaluation_metrics, build_memory_layer_summary

st.set_page_config(page_title="Conversational Memory System", page_icon="🧠", layout="wide")


def _load_bundle() -> Dict[str, Any]:
    cache_key = "memory_bundle"
    if cache_key not in st.session_state:
        pipeline = MemoryPipeline()
        st.session_state[cache_key] = pipeline.build(SAMPLE_DATA_PATH, output_dir=Path("artifacts"))
        st.session_state["chatbot"] = ConversationalMemoryChatbot(st.session_state[cache_key]["retriever"])
    return st.session_state[cache_key]


def _dashboard(bundle: Dict[str, Any]) -> None:
    messages = bundle["messages"]
    topics = bundle["topics"]
    checkpoints = bundle["memory_checkpoints"]
    persona = bundle["persona"]
    memory_layers = build_memory_layer_summary(bundle.get("memory_layers", {}))
    traits = sum(len(items) for items in persona.values())
    persona_rows = [item for values in persona.values() for item in values]

    st.title("Intelligent Conversational Memory System")
    st.caption("Hierarchical memory, topic segmentation, persona extraction, and explainable retrieval.")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Messages", len(messages))
    col2.metric("Total Topics", len(topics))
    col3.metric("Total Checkpoints", len(checkpoints))
    col4.metric("Persona Traits", traits)
    col5.metric("Conversation Stats", f"{len(set(message['speaker'] for message in messages))} speakers")

    topic_df = pd.DataFrame(
        {
            "topic": [topic["topic_id"] for topic in topics],
            "confidence": [topic["confidence"] for topic in topics],
            "messages": [len(topic["messages"]) for topic in topics],
        }
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Topic Confidence")
        st.plotly_chart(px.bar(topic_df, x="topic", y="confidence", color="messages"), use_container_width=True)
    with right:
        st.subheader("Persona Distribution")
        persona_counts = pd.DataFrame({"category": list(persona.keys()), "items": [len(values) for values in persona.values()]})
        st.plotly_chart(px.pie(persona_counts, names="category", values="items"), use_container_width=True)

    st.subheader("Memory Layer Visualization")
    layer_df = pd.DataFrame(memory_layers)
    if not layer_df.empty:
        st.plotly_chart(px.bar(layer_df, x="layer", y="documents", color="confidence", title="Layer Coverage"), use_container_width=True)

    st.subheader("Persona Confidence Distribution")
    confidence_rows = []
    for category, items in persona.items():
        for item in items:
            confidence_rows.append({"category": category, "value": item.get("value"), "confidence": item.get("confidence", 0.0)})
    confidence_df = pd.DataFrame(confidence_rows)
    if not confidence_df.empty:
        st.plotly_chart(px.bar(confidence_df, x="category", y="confidence", color="confidence", title="Persona Confidence"), use_container_width=True)

    st.subheader("Topic Transition Graph")
    transition_rows = []
    for index in range(1, len(topics)):
        transition_rows.append(
            {
                "source": topics[index - 1].get("title", topics[index - 1]["topic_id"]),
                "target": topics[index].get("title", topics[index]["topic_id"]),
                "weight": round((topics[index - 1]["confidence"] + topics[index]["confidence"]) / 2, 4),
            }
        )
    transition_df = pd.DataFrame(transition_rows)
    if not transition_df.empty:
        graph = nx.DiGraph()
        for _, row in transition_df.iterrows():
            graph.add_edge(row["source"], row["target"], weight=row["weight"])
        positions = nx.spring_layout(graph, seed=42)
        edge_x = []
        edge_y = []
        for source, target in graph.edges():
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        node_x = [positions[node][0] for node in graph.nodes()]
        node_y = [positions[node][1] for node in graph.nodes()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#94a3b8"), hoverinfo="none", mode="lines"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(graph.nodes()), textposition="top center", marker=dict(size=16, color="#0f766e")))
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evaluation Metrics")
    metrics = build_evaluation_metrics(bundle)
    metric_df = pd.DataFrame([{"metric": key.replace("_", " ").title(), "score": value} for key, value in metrics.items()])
    st.plotly_chart(px.bar(metric_df, x="metric", y="score", color="score", title="Structural Quality Scores"), use_container_width=True)


def _topic_timeline(bundle: Dict[str, Any]) -> None:
    st.header("Topic Timeline")
    timeline = []
    for topic in bundle["topics"]:
        timeline.append(
            {
                "topic": topic["topic_id"],
                "title": topic.get("title", topic["topic_id"]),
                "start": topic["start_index"],
                "end": topic["end_index"],
                "confidence": topic["confidence"],
            }
        )
    st.dataframe(pd.DataFrame(timeline), use_container_width=True)


def _persona_explorer(bundle: Dict[str, Any]) -> None:
    st.header("Persona Explorer")
    persona = bundle["persona"]
    for category, values in persona.items():
        st.subheader(category.replace("_", " ").title())
        st.write(pd.DataFrame(values))


def _memory_checkpoints(bundle: Dict[str, Any]) -> None:
    st.header("Memory Checkpoints")
    st.dataframe(pd.DataFrame(bundle["memory_checkpoints"]), use_container_width=True)


def _retrieval_inspector(bundle: Dict[str, Any]) -> None:
    st.header("Retrieval Inspector")
    query = st.text_input("Enter a question")
    if st.button("Retrieve") and query:
        result = st.session_state["chatbot"].ask(query)
        st.write("Query")
        st.code(query)
        st.write("Intent")
        st.code(result.get("intent", "general"))

        left, right = st.columns(2)
        with left:
            st.subheader("Retrieved Topic Summaries")
            st.dataframe(pd.DataFrame(result.get("retrieved_topics", [])), use_container_width=True)
            st.subheader("Retrieved Chunks")
            st.dataframe(pd.DataFrame(result.get("retrieved_chunks", [])), use_container_width=True)
        with right:
            st.subheader("Retrieved Persona Facts")
            st.dataframe(pd.DataFrame(result.get("retrieved_persona_facts", [])), use_container_width=True)
            st.subheader("Similarity Scores")
            st.dataframe(pd.DataFrame(result.get("similarity_scores", [])), use_container_width=True)

        st.subheader("Generated Answer")
        st.write(result.get("answer", ""))
        st.subheader("Evidence")
        st.write(result.get("evidence", []))
        st.subheader("Source Attribution Panel")
        st.dataframe(pd.DataFrame(result.get("sources", [])), use_container_width=True)
        st.subheader("Query Explanation Panel")
        explanation = {
            "intent": result.get("intent", "general"),
            "answer_length": len(result.get("answer", "")),
            "evidence_count": len(result.get("evidence", [])),
            "source_count": len(result.get("sources", [])),
        }
        st.json(explanation)


def _chatbot(bundle: Dict[str, Any]) -> None:
    st.header("Chatbot")
    query = st.chat_input("Ask about the user or conversation history")
    if query:
        result = st.session_state["chatbot"].ask(query)
        st.chat_message("assistant").write(result["answer"])
        with st.expander("Evidence"):
            st.write(result.get("evidence", []))
        with st.expander("Sources"):
            st.dataframe(pd.DataFrame(result.get("sources", [])), use_container_width=True)
        with st.expander("Query Explanation"):
            st.json({"intent": result.get("intent", "general"), "source_count": len(result.get("sources", [])), "evidence_count": len(result.get("evidence", []))})


def main() -> None:
    bundle = _load_bundle()
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #f5efe7 0%, #edf3f8 55%, #f8fafc 100%); }
        .block-container { padding-top: 2rem; }
        div[data-testid="metric-container"] { background: rgba(255,255,255,0.72); border-radius: 16px; padding: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Topic Timeline", "Persona Explorer", "Memory Checkpoints", "Retrieval Inspector", "Chatbot"],
    )
    if page == "Dashboard":
        _dashboard(bundle)
    elif page == "Topic Timeline":
        _topic_timeline(bundle)
    elif page == "Persona Explorer":
        _persona_explorer(bundle)
    elif page == "Memory Checkpoints":
        _memory_checkpoints(bundle)
    elif page == "Retrieval Inspector":
        _retrieval_inspector(bundle)
    else:
        _chatbot(bundle)


if __name__ == "__main__":
    main()
