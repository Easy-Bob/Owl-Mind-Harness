import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The browser only talks to this server, so the login cookie works without any cross-site setup.
const apiUrl = process.env.API_URL ?? "http://localhost:8010";
const proxy = Object.fromEntries(
  ["/auth", "/threads", "/runs", "/admin", "/handoffs", "/forum"].map((path) => [path, { target: apiUrl }]),
);

export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
