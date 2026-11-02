import { useEffect, useState, type FormEvent } from "react";
import { api, type User } from "../shared/api";

export function AuthPage({ onLoggedIn }: { onLoggedIn: (user: User) => void }) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [invite, setInvite] = useState("");
  const [accepted, setAccepted] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const signingUp = mode === "signup";

  useEffect(() => {
    api.notice().then((data) => setNotice(data.notice)).catch(() => undefined);
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (signingUp) {
        await api.signup(email, password, invite, accepted);
      }
      onLoggedIn(await api.login(email, password));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth">
      <h1>Owl-Mind</h1>
      <form onSubmit={submit}>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete={signingUp ? "new-password" : "current-password"}
          minLength={signingUp ? 10 : undefined}
          required
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        {signingUp && (
          <>
            <label htmlFor="invite">Invite code</label>
            <input
              id="invite"
              required
              value={invite}
              onChange={(event) => setInvite(event.target.value)}
            />
            <p className="notice">{notice}</p>
            <label className="check">
              <input
                type="checkbox"
                checked={accepted}
                required
                onChange={(event) => setAccepted(event.target.checked)}
              />
              I understand this notice
            </label>
          </>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <button type="submit" disabled={busy}>
          {signingUp ? "Sign up" : "Log in"}
        </button>
      </form>
      <button
        type="button"
        className="link"
        onClick={() => {
          setMode(signingUp ? "login" : "signup");
          setError("");
        }}
      >
        {signingUp ? "I already have an account" : "Create an account"}
      </button>
    </main>
  );
}
