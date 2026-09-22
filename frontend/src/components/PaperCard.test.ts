import { describe, expect, it } from "vitest";

import { formatScore } from "../utils/format";

describe("formatScore", () => {
  it("formats model scores consistently", () => {
    expect(formatScore(0.123456)).toBe("0.1235");
    expect(formatScore(null)).toBe("—");
  });
});
