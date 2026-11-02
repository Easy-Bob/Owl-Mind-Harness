import { useEffect, useState } from "react";
import { AuthPage } from "./experience/AuthPage";
import { ChatPage } from "./experience/ChatPage";
import { api, type User } from "./shared/api";

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);

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
  return <ChatPage user={user} onLoggedOut={() => setUser(null)} />;
}
