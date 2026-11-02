"""Offline: extraction, chunking, the import job and its admin routes."""

import asyncio
import io
import uuid
from dataclasses import replace

import pytest
from pypdf import PdfReader, PdfWriter

from owl_mind.core.config import get_settings
from owl_mind.materials.chunking import (
    OVERLAP_CHARS,
    TARGET_CHARS,
    chunk_sections,
    looks_like_solution,
)
from owl_mind.materials.extract import (
    ExtractionError,
    Section,
    extract,
    extract_isolated,
    file_format,
)
from owl_mind.materials.ingest import SOLUTION_REFUSED, Ingestor
from owl_mind.materials.store import InMemoryMaterialsStore
from tests import material_fixtures as fixtures
from tests.persistence_probe import fake_embed
from tests.test_auth import logged_in
from tests.test_runtime import World

ADMIN_EMAIL = "owner@example.com"
UPLOADER = uuid.UUID(int=7)


class FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model unavailable")
        return fake_embed(texts)


# -- extraction ---------------------------------------------------------------


def test_pdf_keeps_each_page():
    sections = extract(fixtures.pdf(["Bloom filters", "Count-min sketch"]), ".pdf")

    assert [(s.location, s.text) for s in sections] == [
        ("page 1", "Bloom filters"),
        ("page 2", "Count-min sketch"),
    ]


def test_pptx_keeps_slides_titles_tables_and_notes():
    data = fixtures.pptx(
        [("Hashing", "Universal hashing"), ("Sketches", "Count-min")], "Speak slowly"
    )

    sections = extract(data, ".pptx")

    assert [s.location for s in sections] == ["slide 1", "slide 2", "slide 3"]
    assert sections[0].heading == "Hashing"
    assert "Notes: Speak slowly" in sections[0].text
    assert "Bloom filter | O(k)" in sections[2].text


def test_docx_splits_on_headings():
    sections = extract(fixtures.docx([("Tail bounds", "Markov"), ("Chernoff", "Tighter")]), ".docx")

    assert [(s.heading, s.text) for s in sections] == [
        ("Tail bounds", "Markov"),
        ("Chernoff", "Tighter"),
    ]


def test_markdown_splits_on_headings():
    sections = extract(fixtures.MARKDOWN, ".md")

    assert [s.heading for s in sections] == ["Hashing", "Collisions"]


@pytest.mark.parametrize("suffix", [".tex", ".latex"])
def test_latex_keeps_text_and_math_and_drops_commands_and_comments(suffix):
    sections = extract(fixtures.LATEX, suffix)

    text = "\n".join(s.text for s in sections)
    assert [s.heading for s in sections] == ["Tail bounds", "Chernoff"]
    assert "Chernoff bounds are much tighter" in text
    assert "E[X]/a" in text
    assert "comment" not in text
    assert "\\textbf" not in text


@pytest.mark.parametrize(("name", "fix"), [("old.ppt", ".pptx"), ("old.DOC", ".docx")])
def test_legacy_formats_are_refused_with_a_way_forward(name, fix):
    with pytest.raises(ExtractionError, match=fix):
        file_format(name)


@pytest.mark.parametrize("name", ["notes.txt", "archive.zip", "noextension", "x.pdf.exe"])
def test_unsupported_types_are_refused(name):
    with pytest.raises(ExtractionError, match="Unsupported"):
        file_format(name)


@pytest.mark.parametrize("suffix", [".pdf", ".pptx", ".docx"])
def test_damaged_files_are_refused_cleanly(suffix):
    with pytest.raises(ExtractionError):
        extract(b"this is not a real file", suffix)


def test_password_protected_pdf_is_refused():
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(fixtures.pdf(["secret"]))))
    writer.encrypt("pw")
    out = io.BytesIO()
    writer.write(out)

    with pytest.raises(ExtractionError, match="password"):
        extract(out.getvalue(), ".pdf")


def test_pdf_with_no_text_is_refused_as_possibly_scanned():
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    out = io.BytesIO()
    writer.write(out)

    with pytest.raises(ExtractionError, match="scanned"):
        extract(out.getvalue(), ".pdf")


def test_control_characters_are_removed_and_line_breaks_kept():
    sections = extract(b"# Head\x00ing\n\nBloom\x00 filter\x07s\tare\nfast", ".md")

    assert sections[0].heading == "Heading"
    assert sections[0].text == "Bloom filters\tare\nfast"


@pytest.mark.parametrize(
    "credit",
    [
        "Scribe By: Pat Lee (pl12), Sam Roe (sr34)",
        "Scribes: Pat Lee and Sam Roe",
        "scribed by : Pat Lee",
        "SCRIBE: Pat Lee",
    ],
)
def test_scribe_credit_lines_are_removed(credit):
    data = fixtures.pdf([f"Lecture 7 Lecturer: Instructor {credit}", "Bloom filters"])

    text = " ".join(section.text for section in extract(data, ".pdf"))

    assert "Pat" not in text and "Sam" not in text and "pl12" not in text
    assert "Lecture 7" in text
    assert "Bloom filters" in text


def test_names_on_the_lines_after_the_label_are_removed():
    text = (
        "Lecturer: Instructor\nScribe By:\nPat Lee (pl12),\nSam Roe (sr34)\n1 Introduction\nBloom"
    )

    sections = extract(text.encode(), ".md")

    assert "Pat" not in sections[0].text and "Sam" not in sections[0].text
    assert sections[0].text == "1 Introduction\nBloom"


def test_a_netid_on_its_own_is_removed():
    sections = extract(b"Notes by (pl12) on hashing.", ".md")

    assert "pl12" not in sections[0].text
    assert "hashing" in sections[0].text


def test_a_label_with_no_netid_removes_only_its_line():
    sections = extract(b"Scribes: Pat Lee\nBloom filters have false positives.", ".md")

    assert sections[0].text == "Bloom filters have false positives."


def test_a_notes_header_with_unlabeled_names_is_dropped_before_section_one():
    page = "COMP 480 Notes\nLecture 7\nLecturer: Instructor Pat Lee, Sam Roe\n1 Bloom filters\nText"

    sections = extract(fixtures.pdf([page]), ".pdf")

    assert "Pat" not in sections[0].text and "Sam" not in sections[0].text
    assert sections[0].text.startswith("1 Bloom filters")


def test_text_without_a_header_marker_is_kept_whole():
    sections = extract(b"Intro line\n1 First point\nMore", ".md")

    assert sections[0].text == "Intro line\n1 First point\nMore"


def test_ordinary_uses_of_the_word_scribe_are_kept():
    sections = extract(b"# Notes\n\nStudents scribe one lecture each.", ".md")

    assert sections[0].text == "Students scribe one lecture each."


def test_isolated_extraction_gives_the_same_result():
    data = fixtures.pdf(["Bloom filters"])

    assert extract_isolated(data, ".pdf", 30) == extract(data, ".pdf")


def test_isolated_extraction_reports_a_refusal():
    with pytest.raises(ExtractionError, match="couldn't be read"):
        extract_isolated(b"junk", ".pdf", 30)


def test_a_hanging_parser_is_stopped_at_the_time_limit():
    with pytest.raises(ExtractionError, match="too long"):
        extract_isolated(b"x", ".pdf", 1, target=fixtures.slow_child)


def test_a_crashing_parser_is_reported_without_waiting_for_the_limit():
    with pytest.raises(ExtractionError, match="failed"):
        extract_isolated(b"x", ".pdf", 60, target=fixtures.crashing_child)


# -- chunking -----------------------------------------------------------------


def test_chunks_keep_their_place_and_stay_near_the_target_size():
    paragraph = "Hashing spreads keys across slots evenly. " * 20
    sections = [Section("page 1", "Hashing", "\n\n".join([paragraph] * 6))]

    chunks = chunk_sections(sections)

    assert len(chunks) > 1
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.location == "page 1" and chunk.heading == "Hashing" for chunk in chunks)
    assert all(len(chunk.text) <= TARGET_CHARS + OVERLAP_CHARS + 2 for chunk in chunks)


def test_neighboring_chunks_overlap():
    paragraph = " ".join(f"word{index}" for index in range(400))

    first, second = chunk_sections([Section("p", "h", paragraph)])[:2]

    assert second.text.split()[0] in first.text


def test_a_very_long_unbroken_text_is_still_cut():
    chunks = chunk_sections([Section("p", "h", "x" * (TARGET_CHARS * 3))])

    assert all(len(chunk.text) <= TARGET_CHARS + OVERLAP_CHARS + 2 for chunk in chunks)
    assert sum(chunk.text.count("x") for chunk in chunks) >= TARGET_CHARS * 3


@pytest.mark.parametrize(
    "name",
    [
        "hw1_solutions.pdf",
        "Solution.pdf",
        "answer key.docx",
        "Answer_Key_Midterm.pdf",
        "answers.md",
    ],
)
def test_solution_names_are_caught(name):
    assert looks_like_solution(name, [Section("p", "", "notes")])


@pytest.mark.parametrize(
    "name", ["resolution.pdf", "Lecture_3.pdf", "TailBounds.pdf", "answering.md"]
)
def test_ordinary_names_are_not_caught(name):
    assert not looks_like_solution(name, [Section("p", "Bloom filters", "notes")])


def test_a_solution_heading_on_the_first_page_is_caught():
    assert looks_like_solution("hw.pdf", [Section("page 1", "Solutions to Homework 2", "...")])


# -- the import job -----------------------------------------------------------


def run_import(files, embedder=None):
    store = InMemoryMaterialsStore()
    ingestor = Ingestor(store, embedder or FakeEmbedder(), timeout_seconds=30)

    async def scenario():
        jobs = [await ingestor.submit(name, data, UPLOADER) for name, data in files]
        await ingestor.wait()
        return jobs

    jobs = asyncio.run(scenario())
    return store, [store.jobs[job] for job in jobs]


def test_import_saves_chunks_with_vectors_and_raises_the_version():
    store, [job] = run_import([("Lecture_1.pdf", fixtures.pdf(["Bloom filters", "Sketches"]))])

    assert job.status == "done"
    document = store.documents[job.document_id]
    assert document["title"] == "Lecture 1"
    assert [chunk.location for chunk, _ in document["chunks"]] == ["page 1", "page 2"]
    assert all(len(vector) == 384 for _, vector in document["chunks"])
    assert store.version == 1


def test_the_same_content_twice_keeps_one_copy_and_one_version():
    data = fixtures.pdf(["Bloom filters"])

    store, jobs = run_import([("a.pdf", data), ("b.pdf", data)])

    assert [job.status for job in jobs] == ["done", "done"]
    assert jobs[0].document_id == jobs[1].document_id
    assert len(store.documents) == 1
    assert store.version == 1


def test_a_solution_is_refused_and_writes_nothing():
    store, [job] = run_import([("notes.pdf", fixtures.pdf(["Answer key for homework 2"]))])

    assert (job.status, job.reason) == ("failed", SOLUTION_REFUSED)
    assert store.documents == {}
    assert store.version == 0


def test_a_damaged_file_fails_with_a_reason_and_writes_nothing():
    store, [job] = run_import([("broken.pdf", b"junk")])

    assert job.status == "failed"
    assert "couldn't be read" in job.reason
    assert store.documents == {}


def test_an_embedding_failure_leaves_no_partial_document():
    store, [job] = run_import(
        [("notes.pdf", fixtures.pdf(["Bloom filters"]))], embedder=FakeEmbedder(fail=True)
    )

    assert (job.status, job.reason) == ("failed", "The import failed. Try again.")
    assert store.documents == {}


def test_a_refused_type_fails_its_job():
    store, [job] = run_import([("slides.ppt", b"x")])

    assert job.status == "failed"
    assert ".pptx" in job.reason


def test_imports_run_one_at_a_time():
    running, peak = 0, 0

    class CountingEmbedder(FakeEmbedder):
        def embed(self, texts):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                return super().embed(texts)
            finally:
                running -= 1

    files = [(f"n{i}.pdf", fixtures.pdf([f"topic {i}"])) for i in range(3)]
    store, jobs = run_import(files, embedder=CountingEmbedder())

    assert [job.status for job in jobs] == ["done"] * 3
    assert peak == 1


def test_a_restart_fails_unfinished_jobs():
    store = InMemoryMaterialsStore()

    async def scenario():
        await store.create_job("a.pdf", UPLOADER)
        return await store.fail_unfinished_jobs()

    assert asyncio.run(scenario()) == 1
    assert next(iter(store.jobs.values())).reason.startswith("The app restarted")


def test_shutdown_fails_a_running_import():
    store = InMemoryMaterialsStore()

    class SlowEmbedder(FakeEmbedder):
        def embed(self, texts):
            import time

            time.sleep(0.5)
            return super().embed(texts)

    async def scenario():
        ingestor = Ingestor(store, SlowEmbedder(), timeout_seconds=30)
        job = await ingestor.submit("a.pdf", fixtures.pdf(["x"]), UPLOADER)
        await asyncio.sleep(0.3)
        await ingestor.shutdown()
        return job

    job = asyncio.run(scenario())
    assert store.jobs[job].status == "failed"


# -- admin routes --------------------------------------------------------------


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setenv("JWT_SECRET", "s" * 32)
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("CONFIGURED_ADMINS", ADMIN_EMAIL)
    monkeypatch.setenv("INVITE_CODES", "invite-code-1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def world():
    world = World()
    world.app.dependency_overrides.clear()
    world.materials = InMemoryMaterialsStore()
    world.ingestor = Ingestor(world.materials, FakeEmbedder(), timeout_seconds=30)
    world.services = replace(world.services, ingestor=world.ingestor)
    return world


def upload(client, name="Lecture_1.pdf", data=None, declared=True, source_url=None):
    form = {"not_a_solution": "true" if declared else "false"}
    if source_url:
        form["source_url"] = source_url
    return client.post(
        "/admin/materials",
        files={"file": (name, data if data is not None else fixtures.pdf(["Bloom filters"]))},
        data=form,
    )


def finish(world, client):
    client.portal.call(world.ingestor.wait)


def test_admin_uploads_and_the_job_finishes(world):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        response = upload(client, source_url="https://example.edu/Lecture_1.pdf")
        finish(world, client)
        job = client.get(f"/admin/materials/jobs/{response.json()['job_id']}").json()
        listed = client.get("/admin/materials").json()

    assert response.status_code == 202
    assert job["status"] == "done"
    assert listed["corpus_version"] == 1
    assert listed["documents"][0]["source_url"] == "https://example.edu/Lecture_1.pdf"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/admin/materials"),
        ("get", "/admin/materials"),
        ("get", f"/admin/materials/jobs/{uuid.uuid4()}"),
        ("delete", f"/admin/materials/{uuid.uuid4()}"),
    ],
)
def test_students_get_403_and_no_login_gets_401(world, method, path):
    with world.client() as client:
        assert getattr(client, method)(path).status_code == 401
        logged_in(client)
        assert getattr(client, method)(path).status_code == 403

    assert world.materials.jobs == {}


def test_upload_without_the_declaration_is_refused(world):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        response = upload(client, declared=False)

    assert response.status_code == 400
    assert world.materials.jobs == {}


@pytest.mark.parametrize(("name", "status"), [("deck.ppt", 415), ("notes.txt", 415)])
def test_refused_types_get_415_before_any_job(world, name, status):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert upload(client, name=name, data=b"x").status_code == status

    assert world.materials.jobs == {}


def test_oversized_and_empty_files_are_refused(world, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert upload(client, data=b"x" * (1024 * 1024 + 1)).status_code == 413
        assert upload(client, data=b"x" * (1024 * 1024)).status_code == 202
        assert upload(client, data=b"").status_code == 400


def test_delete_removes_a_document_and_unknown_ids_are_404(world):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        job_id = upload(client).json()["job_id"]
        finish(world, client)
        document = client.get(f"/admin/materials/jobs/{job_id}").json()["document_id"]
        assert client.delete(f"/admin/materials/{document}").status_code == 204
        assert client.delete(f"/admin/materials/{document}").status_code == 404
        listed = client.get("/admin/materials").json()

    assert listed == {"corpus_version": 2, "documents": []}


def test_unknown_job_is_404(world):
    with world.client() as client:
        logged_in(client, email=ADMIN_EMAIL)
        assert client.get(f"/admin/materials/jobs/{uuid.uuid4()}").status_code == 404
