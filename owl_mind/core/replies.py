"""Fixed replies, written in code so that they cost no model call."""

from owl_mind.core.intent.taxonomy import Intent, Reply

# What a student can ask, shown in the empty chat too. Fixed text, so it costs no model call.
CAN_HELP_WITH: tuple[tuple[str, str], ...] = (
    ("Explain a concept", "how does a hash table handle collisions?"),
    ("Get unstuck on a problem, with hints", "I can't start this graph problem"),
    ("Check your code's runtime", "what is the runtime of this loop?"),
    ("Quiz you", "give me a short quiz on sorting"),
    ("Plan your revision", "plan two weeks for my algorithms final"),
    ("Find material", "which lecture covers Bloom filters?"),
    ("Reach a human tutor", "I want to talk to a TA"),
)
CANNOT = "I can't answer other subjects, and I won't write full solutions to graded work."


def help_text(topics: list[str]) -> str:
    """What the student can ask, with the topics of the course notes as they are now."""
    lines = ["I can help you study computer science from your course notes:"]
    lines += [f'- {what}: "{example}"' for what, example in CAN_HELP_WITH]
    if topics:
        lines.append("Topics in your course notes: " + ", ".join(topics) + ".")
    lines.append(CANNOT)
    return "\n".join(lines)


# These replies end with help_text, so the student sees what they can ask.
WITH_HELP = (Reply.OFF_TOPIC, Reply.CLARIFY)

TEXT: dict[Reply, str] = {
    Reply.EMPTY: "It looks like that message was empty. What would you like to work on?",
    Reply.GREETING: "Hello! What would you like to study today?",
    Reply.THANKS: "You're welcome. What would you like to do next?",
    Reply.OFF_TOPIC: "That isn't something I can help with.",
    Reply.FEEDBACK: "Thanks for telling me. What would you like to do next?",
    Reply.CLARIFY: "I'm not sure what you need.",
}
HANDOFF_DRAFTED = (
    "I drafted a forum post for a tutor. Check it below, change anything you like, then approve it."
    " Nothing is posted until you do."
)
HANDOFF_UNAVAILABLE = "Posting to the forum isn't available yet, so I can't reach a tutor for you."
AGENTS_UNAVAILABLE = "Tutoring answers are switched off right now. Please try again later."
STUDENT_BUDGET = (
    "You've reached today's limit for tutoring answers. It resets at midnight UTC. You can still"
    " check what's due, or ask for a human tutor on the forum."
)
AGENTS_PAUSED = (
    "Tutoring answers are paused for now. Please try again later, or ask for a human tutor on"
    " the forum."
)
# Put before a tutor's reply when it isn't the first part of a joined answer.
JOIN_LEADS = {
    "concept_tutor": "Here's the idea behind it.",
    "practice_coach": "Now, for the problem you're working on:",
    "study_planner": "And here's how this fits into your studying:",
    "quiz_master": "Now let's check your understanding.",
}
RUN_STOPPED = "I had to stop before finishing this answer. Please try asking again."
RUN_FAILED = "Something went wrong with this answer. Please try again."


def due_topics(topics: list[str]) -> str:
    if not topics:
        return "Nothing is due for review today."
    return "Due for review today: " + ", ".join(topics) + "."


def progress(scheduled: int, due: int) -> str:
    if not scheduled:
        return "You have no topics scheduled for review yet."
    return f"You have {scheduled} topics scheduled for review, and {due} are due today."


def left_out_note(intents: tuple[Intent, ...]) -> str:
    names = ", ".join(intent.value.replace("_", " ") for intent in intents)
    return f"\n\nI only handled part of your message. Please ask again about: {names}."
