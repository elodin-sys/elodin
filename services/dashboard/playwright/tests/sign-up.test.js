import { test, expect } from "@playwright/test";

const usernumber = Math.floor(Math.random() * 1000000);
const username = "playwright+" + usernumber + "@test.com";

test("test", async ({ page }) => {
  await page.goto("https://app.elodin.dev");
  await expect(page.getByRole("img", { name: "Elodin" })).toBeVisible();
  await page.getByRole("link", { name: "Sign up" }).click();
  await expect(page.getByRole("img", { name: "Elodin" })).toBeVisible();
  await page.getByLabel("Email address").click();
  await page.getByLabel("Email address").fill(username);
  await page.getByLabel("Email address").press("Tab");
  await page.getByLabel("Password").fill("1q2w3!Q@W#");
  await expect(
    page.getByRole("button", { name: "Continue", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Continue", exact: true }).click();

  await page.goto("https://app.elodin.dev/onboard/non_commercial");

  await expect(page.getByText("What are you working on?")).toBeVisible();
  await expect(page.getByText("ROCKETS")).toBeVisible();
  await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
  await expect(async () => {
    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByText("Where did you hear about us?")).toBeVisible({
      timeout: 200,
    });
  }).toPass({ timeout: 10000 });

  await expect(page.getByText("LINKEDIN")).toBeVisible();
  await page.getByText("LINKEDIN").click();
  await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
  await expect(async () => {
    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByText("What excites you about Elodin?")).toBeVisible({
      timeout: 200,
    });
  }).toPass({ timeout: 10000 });

  await expect(page.getByText("LIVE 3D VIEWER")).toBeVisible();
  await page.getByText("LIVE 3D VIEWER").click();
  await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
  await expect(async () => {
    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByText("Download & Install the CLI")).toBeVisible({
      timeout: 200,
    });
  }).toPass({ timeout: 10000 });

  await expect(async () => {
    await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByText("Try one of our templates")).toBeVisible({
      timeout: 200,
    });
  }).toPass({ timeout: 10000 });

  await expect(async () => {
    await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByText("Run a Monte Carlo Sim")).toBeVisible({
      timeout: 200,
    });
  }).toPass({ timeout: 10000 });

  await expect(async () => {
    await page.getByRole("img").nth(2).click();
    await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
    await page.getByRole("button", { name: "Next" }).click();
    await page.getByRole("link", { name: "DASHBOARD" }).click();
    await expect(page.getByText("3 Body Problem")).toBeVisible({
      timeout: 200,
    });
  }).toPass({ timeout: 10000 });

  await page.locator("div:nth-child(3) > a").click();
  await page.getByRole("button", { name: "New Sandbox" }).click();
  await page.getByText("Success! Successfully created").click();

  await expect(async () => {
    await expect(page.locator("#editor")).toBeVisible({ timeout: 3000 });
  }).toPass({ timeout: 60000 });

  await new Promise((resolve) => setTimeout(resolve, 10000));

  await expect(page.locator("#editor")).toBeVisible();
  await page
    .locator("li")
    .filter({ hasText: "Share Docs playwright+" + usernumber + "@" })
    .getByRole("img")
    .click();
  await page.getByRole("link", { name: "Log Out" }).click();
  await expect(page.getByRole("img", { name: "Elodin" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Yes" })).toBeVisible();
  await page.getByRole("button", { name: "Yes" }).click();
});
