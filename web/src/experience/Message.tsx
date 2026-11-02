import ReactMarkdown from "react-markdown";

// react-markdown doesn't render raw HTML, and model text is untrusted, so don't add a plugin
// that does. Student text stays plain text.
export function Message({ role, content }: { role: "user" | "assistant"; content: string }) {
  return (
    <article className={`message message-${role}`} aria-label={role === "user" ? "You" : "Owl-Mind"}>
      {role === "user" ? <p>{content}</p> : <ReactMarkdown>{content}</ReactMarkdown>}
    </article>
  );
}
