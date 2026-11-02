"""Importing course materials. Admin only, because every student shares them."""

import uuid
from typing import Annotated, Any

from fastapi import Depends, File, Form, HTTPException, Query, UploadFile

from owl_mind.api.auth import admin_router
from owl_mind.api.deps import CurrentUser, get_services, require_admin
from owl_mind.core.config import get_settings
from owl_mind.core.runtime import Services
from owl_mind.materials.extract import ExtractionError, file_format
from owl_mind.materials.search import MAX_RESULTS

ServicesDep = Annotated[Services, Depends(get_services)]
Admin = Annotated[CurrentUser, Depends(require_admin)]


@admin_router.post("/materials", status_code=202)
async def upload(
    admin: Admin,
    services: ServicesDep,
    file: Annotated[UploadFile, File()],
    not_a_solution: Annotated[bool, Form()] = False,
    source_url: Annotated[str | None, Form(max_length=2000)] = None,
) -> dict[str, Any]:
    if not not_a_solution:
        raise HTTPException(
            status_code=400, detail="Confirm this file isn't a solution or answer key."
        )
    try:
        file_format(file.filename or "")
    except ExtractionError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    limit = get_settings().max_upload_mb * 1024 * 1024
    # Read one byte past the limit, so an oversized file is refused without reading it all.
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="The file is too large.")
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")
    job_id = await services.ingestor.submit(file.filename, data, admin.id, source_url)
    return {"job_id": str(job_id)}


@admin_router.get("/materials/jobs/{job_id}")
async def job(job_id: uuid.UUID, services: ServicesDep) -> dict[str, Any]:
    found = await services.ingestor.store.get_job(job_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": str(found.id),
        "filename": found.filename,
        "status": found.status,
        "reason": found.reason,
        "document_id": str(found.document_id) if found.document_id else None,
    }


@admin_router.get("/materials")
async def documents(services: ServicesDep) -> dict[str, Any]:
    store = services.ingestor.store
    return {
        "corpus_version": await store.corpus_version(),
        "documents": [
            {
                "id": str(document.id),
                "title": document.title,
                "file_type": document.file_type,
                "chunk_count": document.chunk_count,
                "source_url": document.source_url,
            }
            for document in await store.list_documents()
        ],
    }


@admin_router.delete("/materials/{document_id}", status_code=204)
async def remove(document_id: uuid.UUID, services: ServicesDep) -> None:
    if not await services.ingestor.store.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")


@admin_router.get("/materials/search")
async def search(
    services: ServicesDep,
    q: Annotated[str, Query(max_length=500)],
    k: Annotated[int, Query(ge=1, le=MAX_RESULTS)] = 5,
) -> dict[str, Any]:
    """Lets an admin check what tutors would find for a question."""
    if services.search is None:
        raise HTTPException(status_code=503, detail="Search isn't available.")
    results = await services.search.search(q, limit=k)
    return {"results": [vars(result) for result in results]}
