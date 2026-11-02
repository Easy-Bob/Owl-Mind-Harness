"""Routes the virtual paths agents see to where each kind of file really lives."""

from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    FileUploadResponse,
    WriteResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALS_DIR = REPO_ROOT / "materials"
SKILLS_DIR = REPO_ROOT / "skills"
SKILL_SOURCES = ["/skills/"]

_READ_ONLY = "This location is read-only."


class ReadOnlyFilesystemBackend(FilesystemBackend):
    """Refuses every change here too, so a permissions mistake still can't alter these files."""

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=_READ_ONLY)

    def edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        return EditResult(error=_READ_ONLY)

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=_READ_ONLY)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]


def _student_namespace(area: str):
    def namespace(runtime) -> tuple[str, str]:
        # No fallback: a run with no student must fail, never reach a shared namespace.
        return (area, str(runtime.context.student_id))

    return namespace


def build_backend(
    materials_dir: Path = MATERIALS_DIR, skills_dir: Path = SKILLS_DIR
) -> CompositeBackend:
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/materials/": ReadOnlyFilesystemBackend(root_dir=materials_dir, virtual_mode=True),
            "/skills/": ReadOnlyFilesystemBackend(root_dir=skills_dir, virtual_mode=True),
            "/memories/": StoreBackend(namespace=_student_namespace("memories")),
            "/reviews/": StoreBackend(namespace=_student_namespace("reviews")),
            "/scratch/": StateBackend(),
        },
    )
