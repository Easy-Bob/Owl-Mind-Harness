import { useEffect, useState } from "react";
import { api, type ForumAccount } from "../shared/api";

const RESULTS: Record<string, string> = {
  connected: "Your GitHub account is connected. Forum posts will use it.",
  failed: "Connecting your GitHub account didn't work. Nothing was changed.",
};

// Lets a student post to the forum as themself instead of the Owl-Mind bot.
export function ForumAccountControl({ onFailure }: { onFailure: (caught: unknown) => void }) {
  const [account, setAccount] = useState<ForumAccount | null>(null);
  const [result, setResult] = useState("");

  useEffect(() => {
    api
      .forumAccount()
      .then(setAccount)
      .catch(() => setAccount(null));
    // GitHub sends the student back to /?forum=connected or /?forum=failed.
    const outcome = new URLSearchParams(window.location.search).get("forum");
    if (outcome && RESULTS[outcome]) {
      setResult(RESULTS[outcome]);
      window.history.replaceState(null, "", window.location.pathname);
    }
  }, []);

  if (!account?.available) {
    return null;
  }

  async function disconnect() {
    try {
      await api.disconnectForum();
      setAccount(await api.forumAccount());
      setResult("");
    } catch (caught) {
      onFailure(caught);
    }
  }

  return (
    <div className="forum-account" aria-label="Forum account">
      {result && <p role="status">{result}</p>}
      {account.connected ? (
        <>
          <p>
            Forum posts use GitHub account <strong>{account.login}</strong>.
          </p>
          {account.needs_reconnect && (
            <p className="error" role="alert">
              This connection stopped working. Reconnect it, or disconnect to post as the bot.
            </p>
          )}
          <div className="forum-account-actions">
            {account.needs_reconnect && <a href="/forum/account/connect">Reconnect</a>}
            <button type="button" onClick={disconnect}>
              Disconnect
            </button>
          </div>
        </>
      ) : (
        <>
          <p>Forum posts use the Owl-Mind bot.</p>
          <a href="/forum/account/connect">Connect your GitHub account</a>
        </>
      )}
    </div>
  );
}
