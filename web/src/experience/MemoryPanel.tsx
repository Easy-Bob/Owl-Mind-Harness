import { useState } from "react";
import { api, type MemoryView } from "../shared/api";

// What the tutors remember across sessions. The student can read all of it and delete it.
export function MemoryPanel({ onFailure }: { onFailure: (caught: unknown) => void }) {
  const [memory, setMemory] = useState<MemoryView | null>(null);
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);

  async function toggle() {
    if (open) {
      setOpen(false);
      return;
    }
    try {
      setMemory(await api.memory());
      setOpen(true);
    } catch (caught) {
      onFailure(caught);
    }
  }

  async function forget() {
    try {
      await api.deleteMemory();
      setMemory({ profile: null, episodes: [] });
      setConfirming(false);
    } catch (caught) {
      onFailure(caught);
    }
  }

  const profile = memory?.profile;
  const empty = !profile && (memory?.episodes.length ?? 0) === 0;
  return (
    <div className="memory">
      <button type="button" aria-expanded={open} onClick={toggle}>
        What tutors remember
      </button>
      {open && memory && (
        <section aria-label="What tutors remember">
          {empty ? (
            <p>Nothing yet. Tutors remember a session about 30 minutes after it ends.</p>
          ) : (
            <>
              {profile && (
                <ul>
                  {profile.strengths.length > 0 && <li>Strengths: {profile.strengths.join(", ")}</li>}
                  {profile.weak_topics.length > 0 && (
                    <li>Working on: {profile.weak_topics.join(", ")}</li>
                  )}
                  {profile.style && <li>Prefers: {profile.style}</li>}
                </ul>
              )}
              {memory.episodes.length > 0 && (
                <ol className="memory-episodes">
                  {memory.episodes.map((episode) => (
                    <li key={episode.key}>
                      <span className="handoff-note">{episode.written_at.slice(0, 10)}</span>{" "}
                      {episode.summary}
                    </li>
                  ))}
                </ol>
              )}
              {confirming ? (
                <div className="handoff-actions">
                  <button type="button" onClick={() => setConfirming(false)}>
                    Keep it
                  </button>
                  <button type="button" onClick={forget}>
                    Delete everything
                  </button>
                </div>
              ) : (
                <button type="button" onClick={() => setConfirming(true)}>
                  Delete my memory
                </button>
              )}
            </>
          )}
        </section>
      )}
    </div>
  );
}
