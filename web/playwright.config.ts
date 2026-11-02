import { defineConfig } from "@playwright/test";

const webPort = 4173;

// The spec starts and restarts the API itself, because one test restarts it on purpose.
export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  timeout: 60_000,
  // A proxy in the spec forwards this port to the web server.
  use: { baseURL: "http://localhost:4174" },
  webServer: {
    command: `npm run build && npm run preview -- --port ${webPort} --strictPort`,
    url: `http://localhost:${webPort}`,
    env: { API_URL: "http://localhost:8012" },
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
