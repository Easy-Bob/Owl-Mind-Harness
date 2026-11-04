import { useRef, useState } from "react";
import type { Approval } from "../shared/api";

type Props = {
  approval: Approval;
  onAnswer: (approved: boolean) => Promise<void>;
};

// The summary and items come from the student's own saved records, written by the server's code.
export function ApprovalCard({ approval, onAnswer }: Props) {
  // Set straight away, so a double click can't send two answers.
  const busy = useRef(false);
  const [waiting, setWaiting] = useState(false);

  async function answer(approved: boolean) {
    if (busy.current) {
      return;
    }
    busy.current = true;
    setWaiting(true);
    try {
      await onAnswer(approved);
    } finally {
      busy.current = false;
      setWaiting(false);
    }
  }

  return (
    <section className="handoff" aria-label="Approval needed">
      <h2>Your approval is needed</h2>
      <p>{approval.summary}</p>
      {approval.items.length > 0 && (
        <ul className="approval-items">
          {approval.items.map((item) => (
            <li key={item.topic}>
              {item.topic}, review on {item.review_date}
            </li>
          ))}
        </ul>
      )}
      <p className="handoff-note">
        Nothing changes until you choose. This chat waits for your answer, for up to 24 hours.
      </p>
      <div className="handoff-actions">
        <button type="button" disabled={waiting} onClick={() => answer(false)}>
          Reject
        </button>
        <button type="button" disabled={waiting} onClick={() => answer(true)}>
          Approve
        </button>
      </div>
    </section>
  );
}
