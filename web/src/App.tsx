import { useEffect, useState } from "react";
import { AdminPage } from "./admin/AdminPage";
import { AuthPage } from "./experience/AuthPage";
import { ChatPage } from "./experience/ChatPage";
import { api, type User } from "./shared/api";

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);
  const [mode, setMode] = useState<"experience" | "admin">("experience");

  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setChecked(true));
  }, []);

  if (!checked) {
    return null;
  }
  if (user === null) {
    return <AuthPage onLoggedIn={setUser} />;
  }
  const chat = <ChatPage user={user} onLoggedOut={() => setUser(null)} />;
  // The server enforces the role on every /admin route. This only decides what to show.
  if (user.role !== "admin") {
    return chat;
  }
  const inAdmin = mode === "admin";
  return (
    <div className="with-mode-switch">
      <header className="mode-switch">
        <button type="button" onClick={() => setMode(inAdmin ? "experience" : "admin")}>
          {inAdmin ? "Chat" : "Admin"}
        </button>
      </header>
      <div className="mode-body">{inAdmin ? <AdminPage /> : chat}</div>
    </div>
  );
}
