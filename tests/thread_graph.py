"""A small graph with no model, used to exercise checkpoints like a real conversation."""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _reply(state: State) -> State:
    return {"messages": [AIMessage(f"reply {len(state['messages'])}")]}


def build_graph(checkpointer: BaseCheckpointSaver):
    return (
        StateGraph(State)
        .add_node("reply", _reply)
        .add_edge(START, "reply")
        .compile(checkpointer=checkpointer)
    )


def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
