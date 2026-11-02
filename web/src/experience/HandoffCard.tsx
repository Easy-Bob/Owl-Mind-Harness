import { useRef, useState } from "react";
import { api, ApiError, type Handoff, type SimilarPost } from "../shared/api";

type Props = {
  handoff: Handoff;
  similar: SimilarPost[];
  onChange: (handoff: Handoff) => void;
  onFailure: (caught: unknown) => void;
};

// The student sees and approves the exact text before anything reaches the forum.
export function HandoffCard({ handoff, similar, onChange, onFailure }: Props) {
  const [title, setTitle] = useState(handoff.title);
  const [body, setBody] = useState(handoff.body);
  const [graded, setGraded] = useState(handoff.graded);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  // Set straight away, so a double click can't send two approvals.
  const busy = useRef(false);
  const [waiting, setWaiting] = useState(false);

  async function act(action: () => Promise<Handoff>) {
    if (busy.current) {
      return;
    }
    busy.current = true;
    setWaiting(true);
    setError("");
    setNotice("");
    try {
      onChange(await action());
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 409 && caught.data) {
        // Redaction changed the text. Show the new version and ask again.
        const revised = caught.data as Handoff;
        setTitle(revised.title);
        setBody(revised.body);
        setGraded(revised.graded);
        setNotice(
          "Some details were removed or shortened. Check the new text, then approve again.",
        );
      } else if (caught instanceof ApiError && caught.status === 401) {
        onFailure(caught);
      } else {
        setError(caught instanceof Error ? caught.message : "Something went wrong.");
      }
    } finally {
      busy.current = false;
      setWaiting(false);
    }
  }

  if (handoff.status === "cancelled") {
    return null;
  }
  if (handoff.status === "posted") {
    return (
      <section className="handoff" aria-label="Forum post">
        <p>
          Posted to the forum: <a href={handoff.url ?? "#"}>{handoff.title}</a>. A tutor will answer
          there.
        </p>
      </section>
    );
  }
  if (handoff.status === "posting" || handoff.status === "retrying") {
    return (
      <section className="handoff" aria-label="Forum post">
        <p>GitHub didn't answer, so your approved post will be sent again shortly.</p>
      </section>
    );
  }
  if (handoff.status === "fallback") {
    return (
      <section className="handoff" aria-label="Forum post">
        <p>
          The post couldn't be sent. Copy the text below and post it yourself
          {handoff.forum_url ? (
            <>
              {" "}
              on <a href={handoff.forum_url}>the forum</a>
            </>
          ) : null}
          .
        </p>
        <textarea readOnly aria-label="Post to copy" rows={8} value={`${title}\n\n${body}`} />
      </section>
    );
  }

  return (
    <section className="handoff" aria-label="Forum post draft">
      <h2>Draft for the tutor forum</h2>
      <p className="handoff-note">
        Nothing is posted until you approve. Once posted, the question stays on the forum, where
        course staff and other students can read it.
      </p>
      {similar.length > 0 && (
        <div className="handoff-similar">
          <p>Similar questions already on the forum:</p>
          <ul>
            {similar.map((post) => (
              <li key={post.url}>
                <a href={post.url}>{post.title}</a>
              </li>
            ))}
          </ul>
        </div>
      )}
      <label htmlFor={`handoff-title-${handoff.id}`}>Title</label>
      <input
        id={`handoff-title-${handoff.id}`}
        value={title}
        maxLength={120}
        onChange={(event) => setTitle(event.target.value)}
      />
      <label htmlFor={`handoff-body-${handoff.id}`}>Post</label>
      <textarea
        id={`handoff-body-${handoff.id}`}
        value={body}
        maxLength={6000}
        rows={10}
        onChange={(event) => setBody(event.target.value)}
      />
      <label className="handoff-graded">
        <input
          type="checkbox"
          checked={graded}
          onChange={(event) => setGraded(event.target.checked)}
        />
        This is about a graded assignment (only a short excerpt is posted)
      </label>
      {notice && (
        <p className="handoff-notice" role="status">
          {notice}
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div className="handoff-actions">
        <button
          type="button"
          disabled={waiting}
          onClick={() => act(() => api.cancelHandoff(handoff.id))}
        >
          Cancel
        </button>
        <button
          type="button"
          disabled={waiting || !title.trim() || !body.trim()}
          onClick={() => act(() => api.approveHandoff(handoff.id, title, body, graded))}
        >
          Approve and post
        </button>
      </div>
    </section>
  );
}
