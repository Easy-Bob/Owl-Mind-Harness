// A real browser, API, Postgres and Redis, with a scripted model that costs nothing.
// Needs the compose services: docker compose up -d --wait

import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import net from "node:net";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

const repoRoot = path.resolve(import.meta.dirname, "../..");
const apiUrl = "http://localhost:8012";
const password = "correct horse battery";
const invite = "e2e-invite";
const adminEmail = "e2e-admin@example.com";
const env = {
  ...process.env,
  DATABASE_URL: "postgresql://owl:owl_dev@localhost:5433/owl",
  REDIS_URL: "redis://localhost:6380/0",
  DEEPSEEK_API_KEY: "not-a-real-key",
  LIVE_AGENT_RUNS_ENABLED: "true",
  JWT_SECRET: "e".repeat(32),
  COOKIE_SECURE: "false",
  INVITE_CODES: invite,
  CONFIGURED_ADMINS: adminEmail,
};

let api: ChildProcess;
let consumer: ChildProcess;

// The browser reaches the web server through this, so that a test can really cut a live stream.
const proxySockets = new Set<net.Socket>();
const proxy = net.createServer((client) => {
  const upstream = net.connect(4173, "localhost");
  for (const socket of [client, upstream]) {
    proxySockets.add(socket);
    socket.on("close", () => proxySockets.delete(socket));
    socket.on("error", () => undefined);
  }
  client.pipe(upstream);
  upstream.pipe(client);
});

function cutLiveConnections() {
  const count = proxySockets.size;
  proxySockets.forEach((socket) => socket.destroy());
  return count;
}

async function startApi() {
  api = spawn("uv", ["run", "python", "-m", "tests.e2e_server"], {
    cwd: repoRoot,
    env,
    stdio: "inherit",
  });
  await expect
    .poll(() => fetch(`${apiUrl}/auth/notice`).then((r) => r.status, () => 0), { timeout: 30_000 })
    .toBe(200);
}

async function stopApi() {
  const exited = new Promise((resolve) => api.once("exit", resolve));
  api.kill();
  await exited;
}

test.beforeAll(async () => {
  execFileSync("uv", ["run", "alembic", "upgrade", "head"], { cwd: repoRoot, env });
  await startApi();
  consumer = spawn("uv", ["run", "python", "-m", "owl_mind.worker.telemetry"], {
    cwd: repoRoot,
    env,
    stdio: "inherit",
  });
  await new Promise<void>((resolve) => proxy.listen(4174, "127.0.0.1", resolve));
});

test.afterAll(async () => {
  cutLiveConnections();
  proxy.close();
  consumer.kill();
  await stopApi();
});

async function signUp(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Create an account" }).click();
  await page.getByLabel("Email").fill(`user-${randomUUID()}@example.com`);
  await page.getByLabel("Password").fill(password);
  await page.getByLabel("Invite code").fill(invite);
  await page.getByLabel("I understand this notice").check();
  await page.getByRole("button", { name: "Sign up" }).click();
  await expect(page.getByLabel("Your message")).toBeVisible();
}

async function send(page: Page, message: string) {
  await page.getByLabel("Your message").fill(message);
  await page.getByRole("button", { name: "Send" }).click();
}

const sessions = (page: Page) => page.getByRole("navigation", { name: "Sessions" }).getByRole("listitem");
const answers = (page: Page) => page.getByRole("article", { name: "Owl-Mind" });

test("logged-out visitor sees login, and signup shows the notice", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Log in" })).toBeVisible();
  await page.getByRole("button", { name: "Create an account" }).click();
  await expect(page.getByText("stores data in China")).toBeVisible();
});

test("New chat with no message creates no session", async ({ page }) => {
  await signUp(page);
  await page.getByRole("button", { name: "+ New chat" }).click();
  await page.reload();
  await expect(page.getByLabel("Your message")).toBeVisible();
  await expect(sessions(page)).toHaveCount(0);
});

test("first message starts a session at the top, titled from the message", async ({ page }) => {
  await signUp(page);
  await send(page, "first question");
  await expect(answers(page)).toContainText("Answer to: first question");
  await expect(sessions(page)).toHaveCount(1);

  await page.getByRole("button", { name: "+ New chat" }).click();
  await send(page, "second question");
  await expect(answers(page)).toContainText("Answer to: second question");
  await expect(sessions(page).first()).toHaveText("second question");
  await expect(sessions(page).nth(1)).toHaveText("first question");
});

test("clicking a session reopens its history and continues it", async ({ page }) => {
  await signUp(page);
  await send(page, "old question");
  await expect(answers(page)).toContainText("Answer to: old question");
  await page.getByRole("button", { name: "+ New chat" }).click();
  await expect(answers(page)).toHaveCount(0);

  await sessions(page).getByRole("button", { name: "old question" }).click();
  await expect(answers(page)).toContainText("Answer to: old question");
  await send(page, "follow up");
  await expect(answers(page).last()).toContainText("Answer to: follow up");
  await expect(sessions(page)).toHaveCount(1);
});

test("text appears before the run ends, and the input waits", async ({ page }) => {
  await signUp(page);
  await send(page, "slow please");

  await expect(answers(page)).toContainText("word1", { timeout: 15_000 });
  await expect(answers(page)).not.toContainText("word29");
  await expect(page.getByLabel("Your message")).toBeDisabled();
  await expect(page.getByRole("button", { name: "+ New chat" })).toBeDisabled();

  await expect(answers(page)).toContainText("word29", { timeout: 20_000 });
  await expect(page.getByLabel("Your message")).toBeEnabled();
});

test("a dropped stream reconnects with no text lost or repeated", async ({ page }) => {
  await signUp(page);
  const streams: Promise<Record<string, string>>[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/stream")) {
      streams.push(request.allHeaders());
    }
  });
  await send(page, "slow please");
  await expect(answers(page)).toContainText("word3", { timeout: 15_000 });

  expect(cutLiveConnections()).toBeGreaterThan(0);

  await expect(answers(page)).toContainText("word29", { timeout: 30_000 });
  await expect(page.getByLabel("Your message")).toBeEnabled({ timeout: 10_000 });
  const expected = Array.from({ length: 30 }, (_, index) => `word${index}`).join(" ");
  await expect(answers(page).last()).toHaveText(expected);
  expect(streams.length).toBeGreaterThan(1);
  expect((await streams.at(-1))?.["last-event-id"]).toMatch(/^\d+-\d+$/);
});

test("the session and history survive restarting the API and containers", async ({ page }) => {
  await signUp(page);
  await send(page, "keep this");
  await expect(answers(page)).toContainText("Answer to: keep this");

  await stopApi();
  execFileSync("docker", ["compose", "restart", "postgres", "redis"], { cwd: repoRoot });
  execFileSync("docker", ["compose", "up", "-d", "--wait", "postgres", "redis"], { cwd: repoRoot });
  await startApi();

  await page.reload();
  await sessions(page).getByRole("button", { name: "keep this" }).click();
  await expect(answers(page)).toContainText("Answer to: keep this");
  await send(page, "still here");
  await expect(answers(page).last()).toContainText("Answer to: still here");
});

test("model output with HTML doesn't run, and Markdown still renders", async ({ page }) => {
  await signUp(page);
  await send(page, "unsafe please");
  await expect(answers(page)).toContainText("kept");

  await expect(answers(page).locator("strong")).toHaveText("bold");
  await expect(answers(page).locator("script, img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as { pwned?: boolean }).pwned)).toBeUndefined();
});

test("asking for a tutor shows a draft that is posted only after approval", async ({ page }) => {
  await signUp(page);
  const approvals: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().includes("/approve")) {
      approvals.push(request.postData() ?? "");
    }
  });
  await send(page, "I want to talk to a TA about BFS");

  const card = page.getByRole("region", { name: "Forum post draft" });
  await expect(card).toBeVisible();
  await expect(card.getByRole("link", { name: "An older question" })).toBeVisible();
  expect(approvals).toHaveLength(0);

  await card.getByLabel("Title").fill("BFS visits nodes twice");
  await card.getByRole("button", { name: "Approve and post" }).dblclick();

  const posted = page.getByRole("region", { name: "Forum post" });
  await expect(posted.getByRole("link", { name: "BFS visits nodes twice" })).toBeVisible();
  expect(approvals).toHaveLength(1);
  expect(JSON.parse(approvals[0]).title).toBe("BFS visits nodes twice");

  await page.reload();
  await sessions(page).first().getByRole("button").click();
  await expect(page.getByRole("region", { name: "Forum post" })).toContainText("Posted to the forum");
});

test("a forum reply shows on its session with a badge, and its HTML doesn't run", async ({ page }) => {
  await signUp(page);
  await send(page, "I want to talk to a TA about heaps");
  const card = page.getByRole("region", { name: "Forum post draft" });
  await expect(card).toContainText("Posts as the Owl-Mind bot.");
  const approved = page.waitForResponse((response) => response.url().includes("/approve"));
  await card.getByRole("button", { name: "Approve and post" }).click();
  const handoffId = (await (await approved).json()).id;
  await expect(page.getByRole("region", { name: "Forum post" })).toContainText("Posted to the forum");

  execFileSync("uv", ["run", "python", "-m", "tests.e2e_seed_reply", handoffId], { cwd: repoRoot, env });
  await page.reload();

  const session = sessions(page).first().getByRole("button");
  await expect(session).toContainText("1 new");
  await session.click();
  const replies = page.getByRole("list", { name: "Forum replies" });
  await expect(replies).toContainText("ta_jane");
  await expect(replies).toContainText("Accepted answer");
  await expect(replies.locator("strong").filter({ hasText: "queue" })).toBeVisible();
  await expect(replies.locator("img, script")).toHaveCount(0);
  expect(await page.evaluate(() => (window as { pwned?: boolean }).pwned)).toBeUndefined();
  await expect(session).not.toContainText("new");
});

test("a double click on Send starts one run", async ({ page }) => {
  await signUp(page);
  const starts: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/threads")) {
      starts.push(request.url());
    }
  });
  await page.getByLabel("Your message").fill("only once");
  await page.getByRole("button", { name: "Send" }).dblclick();
  await expect(answers(page)).toContainText("Answer to: only once");

  expect(starts).toHaveLength(1);
  await expect(page.getByRole("article", { name: "You" })).toHaveCount(1);
});

test("the login token can't be read by JavaScript or found in browser storage", async ({ page, context }) => {
  await signUp(page);

  const cookie = (await context.cookies()).find((item) => item.name === "owl_session");
  expect(cookie?.httpOnly).toBe(true);
  expect(await page.evaluate(() => document.cookie)).not.toContain("owl_session");
  expect(
    await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage })),
  ).toBe("{}");
});

test("a student sees no Admin mode control, and logout returns to login", async ({ page }) => {
  await signUp(page);
  await expect(page.getByText("Admin")).toHaveCount(0);

  await page.getByRole("button", { name: "Log out" }).click();
  await expect(page.getByRole("button", { name: "Log in" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("button", { name: "Log in" })).toBeVisible();
});

test("the layout fits a phone with no sideways scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await signUp(page);
  await send(page, "a question on a phone with a long-enough-word-to-test-wrapping-in-the-layout");
  await expect(answers(page)).toContainText("Answer to:");

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBe(0);
  await expect(page.getByRole("button", { name: "+ New chat" })).toBeHidden();
  await page.getByRole("button", { name: "Sessions" }).click();
  await expect(page.getByRole("button", { name: "+ New chat" })).toBeVisible();
});

test("the newest text stays in view as an answer streams", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 420 });
  await signUp(page);
  for (const message of ["first", "second", "third"]) {
    await send(page, message);
    await expect(answers(page).last()).toContainText(`Answer to: ${message}`);
  }
  await send(page, "slow please");

  await expect(answers(page).last()).toContainText("word29", { timeout: 20_000 });
  await expect(answers(page).last().getByText("word29")).toBeInViewport();
});

test("the chat can be used with only the keyboard", async ({ page }) => {
  await signUp(page);
  await page.getByLabel("Your message").focus();
  await page.keyboard.type("typed only");
  await page.keyboard.press("Enter");
  await expect(answers(page)).toContainText("Answer to: typed only");

  await page.getByLabel("Your message").focus();
  await page.keyboard.type("line one");
  await page.keyboard.press("Shift+Enter");
  await page.keyboard.type("line two");
  await expect(page.getByLabel("Your message")).toHaveValue("line one\nline two");
});

test("an admin sees a student's request, tokens and timeline, with no message text", async ({ page, browser }) => {
  const studentContext = await browser.newContext({ baseURL: "http://localhost:4174" });
  const student = await studentContext.newPage();
  await signUp(student);
  const email = await student.getByRole("navigation", { name: "Sessions" }).getByText("@example.com").textContent();
  const secret = `secret-message-${randomUUID()}`;
  await send(student, secret);
  await expect(answers(student)).toContainText("Answer to:");
  await send(student, "hi");
  await expect(answers(student).last()).toContainText("Hello");

  // A second signup attempt on later runs just answers 409.
  await page.request.post("/auth/signup", {
    data: { email: adminEmail, password, invite_code: invite, accepted_notice: true },
  });
  await page.goto("/");
  await page.getByLabel("Email").fill(adminEmail);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Log in" }).click();
  await page.getByRole("button", { name: "Admin" }).click();

  await page.getByLabel("Search by email or run ID").fill(email!);
  await page.getByRole("button", { name: "Search" }).click();
  const rows = page.getByRole("row");
  await expect(rows).toHaveCount(3);
  await expect(rows.nth(1)).toContainText("template");
  await expect(rows.nth(2)).toContainText("orchestrated");
  await expect(page.locator("body")).not.toContainText(secret);

  // The consumer writes model calls just after a run, so reopen until they're there.
  await expect(async () => {
    await rows.nth(2).getByRole("button", { name: "Open" }).click();
    try {
      await expect(page.getByRole("cell", { name: "orchestrator" }).first()).toBeVisible({ timeout: 1000 });
    } catch (error) {
      await page.getByRole("button", { name: "Back to requests" }).click();
      throw error;
    }
  }).toPass({ timeout: 30_000 });
  await expect(page.getByText("recorded in the access log")).toBeVisible();
  await expect(page.getByRole("cell", { name: "intent" })).toBeVisible();
  await expect(page.getByText("usage", { exact: true })).toBeVisible();
  await expect(page.locator("body")).not.toContainText(secret);

  await page.getByRole("button", { name: "Back to requests" }).click();
  await page.getByRole("button", { name: "Model calls" }).click();
  await expect(page.getByRole("heading", { name: "By component" })).toBeVisible();
  await expect(page.getByRole("cell", { name: "orchestrator" }).first()).toBeVisible();

  await page.getByRole("button", { name: "Chat" }).click();
  await expect(page.getByLabel("Your message")).toBeVisible();
  await studentContext.close();
});

test("an admin sees a handoff's steps and the post link, with none of the student's text", async ({ page, browser }) => {
  const studentContext = await browser.newContext({ baseURL: "http://localhost:4174" });
  const student = await studentContext.newPage();
  await signUp(student);
  const email = await student.getByRole("navigation", { name: "Sessions" }).getByText("@example.com").textContent();
  const secret = `secret-draft-${randomUUID()}`;
  await send(student, `I want to talk to a TA about ${secret}`);
  const card = student.getByRole("region", { name: "Forum post draft" });
  await expect(card.getByLabel("Post", { exact: true })).toHaveValue(new RegExp(secret));
  await card.getByRole("button", { name: "Approve and post" }).click();
  await expect(student.getByRole("region", { name: "Forum post" })).toContainText("Posted to the forum");

  await page.request.post("/auth/signup", {
    data: { email: adminEmail, password, invite_code: invite, accepted_notice: true },
  });
  await page.goto("/");
  await page.getByLabel("Email").fill(adminEmail);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Log in" }).click();
  await page.getByRole("button", { name: "Admin" }).click();
  await page.getByLabel("Search by email or run ID").fill(email!);
  await page.getByRole("button", { name: "Search" }).click();
  await page.getByLabel("Route").selectOption("handoff");

  const rows = page.getByRole("row");
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(1)).toContainText("handoff: posted");
  await rows.nth(1).getByRole("button", { name: "Open" }).click();
  const timeline = page.getByRole("list").last();
  await expect(timeline).toContainText("handoff drafted");
  await expect(timeline).toContainText("handoff approved");
  await expect(timeline).toContainText("handoff posted: posted as: bot");
  await expect(timeline.getByRole("link", { name: "forum post" })).toHaveAttribute("href", /\/discussions\/\d+$/);
  await expect(page.locator("body")).not.toContainText(secret);
  await studentContext.close();
});

test("a student is refused admin data even when asking directly", async ({ page }) => {
  await signUp(page);
  for (const path of ["/admin/requests", "/admin/model-calls", "/admin/me"]) {
    expect((await page.request.get(path)).status()).toBe(403);
  }
  await expect(page.getByRole("button", { name: "Admin" })).toHaveCount(0);
});
