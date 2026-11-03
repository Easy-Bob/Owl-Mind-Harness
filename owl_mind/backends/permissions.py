"""What each agent's file tools can reach.

DeepAgents allows any path that matches no rule, so every list ends by denying everything.
"""

from deepagents.middleware.filesystem import FilesystemPermission

from owl_mind.core.intent.taxonomy import Subagent


def _rules(
    *, read: tuple[str, ...] = (), write: tuple[str, ...] = ()
) -> list[FilesystemPermission]:
    def paths(roots: tuple[str, ...]) -> list[str]:
        return [path for root in roots for path in (root, f"{root}/**")]

    rules = []
    if write:
        rules.append(FilesystemPermission(operations=["write"], paths=paths(write)))
    # Being able to write a path means being able to read back what was written.
    if read or write:
        rules.append(FilesystemPermission(operations=["read"], paths=paths(read + write)))
    rules.append(FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"))
    return rules


ORCHESTRATOR_PERMISSIONS = _rules(read=("/memories", "/skills"), write=("/scratch",))

TUTOR_PERMISSIONS: dict[Subagent, list[FilesystemPermission]] = {
    Subagent.CONCEPT_TUTOR: _rules(read=("/skills",)),
    Subagent.PRACTICE_COACH: _rules(read=("/skills",)),
    Subagent.STUDY_PLANNER: _rules(read=("/memories", "/skills")),
    Subagent.QUIZ_MASTER: _rules(read=("/skills",), write=("/scratch/quiz",)),
}
