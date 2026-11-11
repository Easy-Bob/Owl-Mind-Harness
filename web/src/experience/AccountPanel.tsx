import { type FormEvent, useState } from "react";
import { api, ApiError } from "../shared/api";

export const DELETE_CONFIRMATION = "delete my account";

// Deleting an account removes the student's data here. Forum posts already on GitHub stay.
export function AccountPanel({ onDeleted }: { onDeleted: () => void }) {
  const [open, setOpen] = useState(false);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function remove(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.deleteAccount(password, confirm);
      onDeleted();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Something went wrong.");
      setBusy(false);
    }
  }

  return (
    <div className="account">
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
        Delete my account
      </button>
      {open && (
        <form aria-label="Delete my account" onSubmit={remove}>
          <p>
            This removes your chats, what tutors remember, your plans, reviews and quizzes, and
            your forum drafts here. It can't be undone. Posts already on the forum stay there.
          </p>
          <label>
            Password
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          </label>
          <label>
            Type "{DELETE_CONFIRMATION}"
            <input value={confirm} onChange={(e) => setConfirm(e.target.value)} />
          </label>
          {error && <p role="alert">{error}</p>}
          <button
            type="submit"
            disabled={busy || !password || confirm.trim().toLowerCase() !== DELETE_CONFIRMATION}
          >
            Delete everything
          </button>
        </form>
      )}
    </div>
  );
}
