import { Suspense } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, test } from "vitest";
import { lazyView } from "./lazyView";

const Greeting = ({ name }: { name: string }) => <p>Hello {name}</p>;

const shown = (View: typeof Greeting) =>
  renderToStaticMarkup(
    <Suspense fallback={<p>Opening…</p>}>
      <View name="tensor" />
    </Suspense>,
  );

test("a view that has not arrived waits behind its fallback", () => {
  const View = lazyView(() => Promise.resolve(Greeting));
  expect(shown(View)).toBe("<p>Opening…</p>");
});

test("a view that has arrived renders at once, without suspending", async () => {
  const View = lazyView(() => Promise.resolve(Greeting));
  await View.preload();
  expect(shown(View)).toBe("<p>Hello tensor</p>");
});

test("a failed fetch is tried again", async () => {
  let tries = 0;
  const View = lazyView(() =>
    ++tries === 1
      ? Promise.reject(new Error("offline"))
      : Promise.resolve(Greeting),
  );
  await expect(View.preload()).rejects.toThrow("offline");
  await View.preload();
  expect(tries).toBe(2);
  expect(shown(View)).toContain("Hello");
});
