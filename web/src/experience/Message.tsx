import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, type Citation } from "../shared/api";

const CITE = /\[cite:([^\]\s]+)\]/g;

// Each distinct citation gets a number, in the order it first appears.
export function numberCitations(content: string): { text: string; ids: string[] } {
  const ids: string[] = [];
  const text = content.replace(CITE, (_, id: string) => {
    if (!ids.includes(id)) {
      ids.push(id);
    }
    return `[${ids.indexOf(id) + 1}]`;
  });
  return { text, ids };
}

function Sources({ ids, final }: { ids: string[]; final: boolean }) {
  const [found, setFound] = useState<Record<string, Citation | null> | null>(null);
  const [failed, setFailed] = useState(false);
  const key = ids.join(",");

  useEffect(() => {
    // Looked up once the answer is complete, so a streaming answer makes one request.
    if (!final || !key) {
      return;
    }
    let current = true;
    api
      .citations(key.split(","))
      .then((result) => current && setFound(result))
      .catch(() => current && setFailed(true));
    return () => {
      current = false;
    };
  }, [key, final]);

  if (ids.length === 0) {
    return null;
  }
  if (failed) {
    return (
      <p className="sources" aria-label="Sources">
        Sources unavailable right now.
      </p>
    );
  }
  return (
    <ol className="sources" aria-label="Sources">
      {ids.map((id) => {
        const source = found?.[id];
        if (found && !source) {
          return <li key={id}>source not found</li>;
        }
        if (!source) {
          return <li key={id}>…</li>;
        }
        const label = source.location ? `${source.title}, ${source.location}` : source.title;
        return (
          <li key={id}>
            {source.url ? (
              <a href={source.url} target="_blank" rel="noreferrer">
                {label}
              </a>
            ) : (
              label
            )}
          </li>
        );
      })}
    </ol>
  );
}

// react-markdown doesn't render raw HTML, and model text is untrusted, so don't add a plugin
// that does. Student text stays plain text.
export function Message({
  role,
  content,
  final = true,
}: {
  role: "user" | "assistant";
  content: string;
  final?: boolean;
}) {
  if (role === "user") {
    return (
      <article className="message message-user" aria-label="You">
        <p>{content}</p>
      </article>
    );
  }
  const { text, ids } = numberCitations(content);
  return (
    <article className="message message-assistant" aria-label="Owl-Mind">
      <ReactMarkdown>{text}</ReactMarkdown>
      <Sources ids={ids} final={final} />
    </article>
  );
}
