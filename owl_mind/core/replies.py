"""Fixed replies, written in code so that they cost no model call."""

from owl_mind.core.intent.taxonomy import Intent, Reply

TEXT: dict[Reply, str] = {
    Reply.EMPTY: "It looks like that message was empty. What would you like to work on?",
    Reply.GREETING: "Hello! What would you like to study today?",
    Reply.THANKS: "You're welcome. What would you like to do next?",
    Reply.OFF_TOPIC: "I can only help with studying computer science. What are you working on?",
    Reply.PROGRESS: "Progress reports aren't available yet.",
    Reply.FEEDBACK: "Thanks for telling me. What would you like to do next?",
    Reply.CLARIFY: "I'm not sure what you need. Could you say a little more about it?",
}
HANDOFF_UNAVAILABLE = "Posting to the forum isn't available yet, so I can't reach a tutor for you."
AGENTS_UNAVAILABLE = "Tutoring answers aren't available yet. This feature is still being built."
RUN_STOPPED = "I had to stop before finishing this answer. Please try asking again."
RUN_FAILED = "Something went wrong with this answer. Please try again."


def due_topics(topics: list[str]) -> str:
    if not topics:
        return "Nothing is due for review today."
    return "Due for review today: " + ", ".join(topics) + "."


def left_out_note(intents: tuple[Intent, ...]) -> str:
    names = ", ".join(intent.value.replace("_", " ") for intent in intents)
    return f"\n\nI only handled part of your message. Please ask again about: {names}."
