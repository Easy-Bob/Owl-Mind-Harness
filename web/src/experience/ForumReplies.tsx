import ReactMarkdown from "react-markdown";
import type { ForumReply } from "../shared/api";

// Replies are other people's text from the forum. react-markdown doesn't render raw HTML, so
// don't add a plugin that does.
export function ForumReplies({ replies }: { replies: ForumReply[] }) {
  if (replies.length === 0) {
    return <p className="handoff-note">No replies yet. They'll appear here when they arrive.</p>;
  }
  return (
    <ol className="forum-replies" aria-label="Forum replies">
      {replies.map((reply) => (
        <li key={reply.id} className={reply.is_answer ? "forum-answer" : undefined}>
          <p className="forum-reply-meta">
            <span>{reply.author}</span>
            {reply.is_answer && <strong> Accepted answer</strong>}
            {reply.new && <span className="forum-new"> New</span>}
          </p>
          <ReactMarkdown>{reply.body}</ReactMarkdown>
        </li>
      ))}
    </ol>
  );
}

export function postAsLabel(postAs: string | null): string {
  return !postAs || postAs === "bot" ? "the Owl-Mind bot" : postAs;
}
