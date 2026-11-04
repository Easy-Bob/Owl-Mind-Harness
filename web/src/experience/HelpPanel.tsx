import { useEffect, useState } from "react";
import { api, type Help } from "../shared/api";

// What a student can ask, and the course notes' topics. The server reads both from its records,
// with no model call, and the topics change whenever a course file is imported or deleted.
export function HelpPanel() {
  const [help, setHelp] = useState<Help | null>(null);

  useEffect(() => {
    let current = true;
    api
      .help()
      .then((loaded) => current && setHelp(loaded))
      .catch(() => current && setHelp(null));
    return () => {
      current = false;
    };
  }, []);

  return (
    <section className="help" aria-label="What you can ask">
      <p className="empty">Ask a question to start studying.</p>
      {help && (
        <>
          <ul>
            {help.can_help_with.map((item) => (
              <li key={item.what}>
                {item.what}: <q>{item.example}</q>
              </li>
            ))}
          </ul>
          {help.topics.length > 0 && (
            <p>
              <strong>Topics in your course notes:</strong> {help.topics.join(", ")}
            </p>
          )}
          <p className="handoff-note">{help.cannot}</p>
        </>
      )}
    </section>
  );
}
