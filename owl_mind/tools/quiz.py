"""post_quiz: multiple-choice quizzes, laid out by code with the answers after the last question.

The quiz master gives the questions, options and correct letters in one call. Code checks the
structure, writes the quiz, and the student sees exactly that text: no model rewrites it, and
the answers cost no extra call. Owner's decision: quizzes aren't graded, the student checks
against the answers.
"""

import re
import string
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from owl_mind.tools.context import RunContext

# Owner's limits: up to 5 questions, each with exactly 4 options and 1 to 3 correct letters.
MAX_QUESTIONS = 5
OPTIONS = 4
MIN_ANSWERS, MAX_ANSWERS = 1, 3
LETTERS = string.ascii_uppercase[:OPTIONS]
# Code adds this marker itself, so the model's own wording of it is removed first.
_CHOOSE_ALL = re.compile(r"[\s(]*(choose|select) all( that apply)?[\s.):]*", re.IGNORECASE)


class QuizQuestion(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    options: list[str] = Field(description="Exactly 4 choices, in order: A, B, C, D.")
    answer: list[str] = Field(description="1 to 3 correct letters. More than one for choose-all.")


def _problem(number: int, question: QuizQuestion) -> str | None:
    options = [" ".join(option.split()) for option in question.options]
    if len(options) != OPTIONS:
        return f"Question {number} needs exactly {OPTIONS} options."
    if any(not option or len(option) > 200 for option in options):
        return f"Question {number} has an empty option or one over 200 characters."
    if len({option.lower() for option in options}) != len(options):
        return f"Question {number} has the same option twice."
    answer = [letter.strip().upper() for letter in question.answer]
    if not MIN_ANSWERS <= len(answer) <= MAX_ANSWERS or len(set(answer)) != len(answer):
        return f"Question {number} needs {MIN_ANSWERS} to {MAX_ANSWERS} different answer letters."
    if any(letter not in LETTERS for letter in answer):
        return f"Question {number} has an answer letter that isn't one of its options."
    return None


def render(topic: str, questions: list[QuizQuestion]) -> str:
    """The quiz as the student sees it: numbered questions, then every answer at the end."""
    lines = [f"**Quiz: {' '.join(topic.split())}**", ""]
    answers = []
    for number, question in enumerate(questions, start=1):
        letters = sorted(letter.strip().upper() for letter in question.answer)
        several = " (choose all that apply)" if len(letters) > 1 else ""
        text = _CHOOSE_ALL.sub(" ", " ".join(question.text.split())).strip()
        text = re.sub(r"\s+([?.!:,])", r"\1", text)
        lines.append(f"{number}. {text}{several}")
        for letter, option in zip(LETTERS, question.options, strict=False):
            lines.append(f"   {letter}. {' '.join(option.split())}")
        lines.append("")
        answers.append(f"{number}. {''.join(letters)}")
    lines.append("**Answers:** " + "  ".join(answers))
    return "\n".join(lines)


@tool
async def post_quiz(
    topic: str, questions: list[QuizQuestion], runtime: ToolRuntime[RunContext]
) -> dict[str, Any]:
    """Post this turn's one multiple-choice quiz, of 1 to 5 questions: each with exactly 4
    options in order (A to D) and 1 to 3 correct letters, more than one for a choose-all
    question. Code shows it with the answers at the end, so don't write the questions or
    answers in your reply."""
    if runtime.context.posted_quizzes:
        return {"error": "This turn's quiz is already posted. Reply with one short line."}
    if not questions:
        return {"error": f"A quiz needs 1 to {MAX_QUESTIONS} questions."}
    # Cut, not refused: a refusal makes the model resend the whole quiz, which costs a call.
    kept = questions[:MAX_QUESTIONS]
    for number, question in enumerate(kept, start=1):
        if problem := _problem(number, question):
            return {"error": problem}
    quiz = render(topic, kept)
    context = runtime.context
    context.posted_quizzes.append(quiz)
    if context.run_id is not None:
        # Kept, so a run resumed by another worker still shows the quiz.
        await context.store.save_quiz_post(context.student_id, context.run_id, topic, quiz)
    result: dict[str, Any] = {"status": "posted", "questions": len(kept)}
    if len(questions) > len(kept):
        result["note"] = f"Only the first {MAX_QUESTIONS} questions were kept."
    return result
