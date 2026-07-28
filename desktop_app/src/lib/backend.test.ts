import { beforeEach, describe, expect, it, vi } from "vitest";

const tauri = vi.hoisted(() => ({
  invoke: vi.fn(),
  handlers: new Map<string, (event: { payload: unknown }) => void>(),
  listen: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({ invoke: tauri.invoke }));
vi.mock("@tauri-apps/api/event", () => ({ listen: tauri.listen }));

import {
  BackendRequestError,
  DEFAULT_REQUEST_TIMEOUT_MS,
  REPLAY_ANALYSIS_TIMEOUT_MS,
  TauriMobileBackend,
  TauriSidecarBackend,
  backendForPlatform,
  requestTimeoutMs,
  sidecarBackend,
} from "./backend";

beforeEach(() => {
  tauri.handlers.clear();
  tauri.invoke.mockReset();
  tauri.listen.mockReset();
  tauri.listen.mockImplementation(async (name: string, handler: (event: { payload: unknown }) => void) => {
    tauri.handlers.set(name, handler);
    return () => tauri.handlers.delete(name);
  });
  tauri.invoke.mockResolvedValue(undefined);
});

async function writtenRequest(): Promise<{ id: number; command: string }> {
  await vi.waitFor(() => expect(tauri.invoke).toHaveBeenCalledWith("write_sidecar", expect.anything()));
  const call = tauri.invoke.mock.calls.find(([command]) => command === "write_sidecar")!;
  return JSON.parse(call[1].line) as { id: number; command: string };
}

async function writtenRequestAt(index: number): Promise<{ id: number; command: string }> {
  await vi.waitFor(() => {
    expect(tauri.invoke.mock.calls.filter(([command]) => command === "write_sidecar")).toHaveLength(index + 1);
  });
  const calls = tauri.invoke.mock.calls.filter(([command]) => command === "write_sidecar");
  return JSON.parse(calls[index][1].line) as { id: number; command: string };
}

describe("Tauri JSONL sidecar client", () => {
  it("keeps long replay analysis requests alive independently of normal commands", () => {
    expect(requestTimeoutMs("game.move")).toBe(DEFAULT_REQUEST_TIMEOUT_MS);
    expect(requestTimeoutMs("replay.analyze")).toBe(REPLAY_ANALYSIS_TIMEOUT_MS);
    expect(REPLAY_ANALYSIS_TIMEOUT_MS).toBeNull();
  });

  it("starts Rust-owned sidecar and resolves a matching JSON response", async () => {
    const backend = new TauriSidecarBackend();
    const start = backend.start();
    const request = await writtenRequest();
    expect(request.command).toBe("system.initialize");
    tauri.handlers.get("sidecar-stdout")?.({
      payload: JSON.stringify({ v: 1, type: "response", id: request.id, ok: true, result: { protocol_version: 1 } }),
    });

    await expect(start).resolves.toMatchObject({ protocol_version: 1 });
    expect(tauri.invoke).toHaveBeenCalledWith("start_sidecar");
    await backend.close();
    expect(tauri.invoke).toHaveBeenCalledWith("stop_sidecar");
  });

  it("maps sidecar error envelopes to typed request errors", async () => {
    const backend = new TauriSidecarBackend();
    const start = backend.start();
    const request = await writtenRequest();
    tauri.handlers.get("sidecar-stdout")?.({
      payload: JSON.stringify({ v: 1, type: "response", id: request.id, ok: false, error: { code: "MODEL_UNAVAILABLE", message: "not ready" } }),
    });

    await expect(start).rejects.toMatchObject({ code: "MODEL_UNAVAILABLE", message: "not ready" } satisfies Partial<BackendRequestError>);
    await backend.close();
  });

  it("surfaces a structured fatal startup event", async () => {
    const backend = new TauriSidecarBackend();
    const start = backend.start();
    await writtenRequest();
    tauri.handlers.get("sidecar-stdout")?.({
      payload: JSON.stringify({
        v: 1,
        type: "event",
        event: "backend.fatal",
        data: { code: "MODEL_REGISTRY_INVALID", message: "registry is missing", details: { path: "model_registry.json" } },
      }),
    });

    await expect(start).rejects.toMatchObject({
      code: "MODEL_REGISTRY_INVALID",
      message: "registry is missing",
      details: { path: "model_registry.json" },
    } satisfies Partial<BackendRequestError>);
    await backend.close();
    expect(tauri.invoke).toHaveBeenCalledWith("stop_sidecar");
  });

  it("formats an object sidecar-exit payload instead of hiding it", async () => {
    const backend = new TauriSidecarBackend();
    const start = backend.start();
    await writtenRequest();
    tauri.handlers.get("sidecar-exit")?.({ payload: { code: 7, signal: null } });

    await expect(start).rejects.toThrow('{"code":7,"signal":null}');
    await backend.close();
  });

  it("serializes a restart behind sidecar shutdown", async () => {
    const backend = new TauriSidecarBackend();
    const firstStart = backend.start();
    const firstRequest = await writtenRequestAt(0);
    tauri.handlers.get("sidecar-stdout")?.({
      payload: JSON.stringify({ v: 1, type: "response", id: firstRequest.id, ok: true, result: { protocol_version: 1 } }),
    });
    await firstStart;

    let releaseStop: (() => void) | undefined;
    let stopCalls = 0;
    tauri.invoke.mockImplementation((command: string) => {
      if (command === "stop_sidecar" && stopCalls++ === 0) {
        return new Promise<void>((resolve) => {
          releaseStop = resolve;
        });
      }
      return Promise.resolve(undefined);
    });

    const closing = backend.close();
    await vi.waitFor(() => expect(releaseStop).toBeTypeOf("function"));
    const restarting = backend.start();
    await Promise.resolve();
    expect(tauri.invoke.mock.calls.filter(([command]) => command === "start_sidecar")).toHaveLength(1);

    releaseStop?.();
    await closing;
    await vi.waitFor(() => {
      expect(tauri.invoke.mock.calls.filter(([command]) => command === "start_sidecar")).toHaveLength(2);
    });
    const secondRequest = await writtenRequestAt(1);
    tauri.handlers.get("sidecar-stdout")?.({
      payload: JSON.stringify({ v: 1, type: "response", id: secondRequest.id, ok: true, result: { protocol_version: 1 } }),
    });

    await expect(restarting).resolves.toMatchObject({ protocol_version: 1 });
    await backend.close();
  });
});

describe("Tauri Android plugin client", () => {
  it("selects the native backend for an Android bundle", () => {
    expect(backendForPlatform("android")).toBeInstanceOf(TauriMobileBackend);
    expect(backendForPlatform("windows")).toBe(sidecarBackend);
    expect(backendForPlatform(undefined)).toBe(sidecarBackend);
  });

  it("initializes through one native plugin request and unwraps the result", async () => {
    const initialized = {
      backend_version: "0.1.0-android",
      protocol_version: 1,
      board: { layers: 6, size: 5, connect_n: 4 },
      mcts_options: [32, 64, 128, 256, 512],
      models: [],
      capabilities: { replay: false },
      state: {},
    };
    tauri.invoke.mockResolvedValue({ ok: true, result: initialized });
    const backend = new TauriMobileBackend();

    await expect(backend.start()).resolves.toEqual(initialized);
    expect(tauri.invoke).toHaveBeenCalledWith("plugin:cubesprite-mobile|request", {
      command: "system.initialize",
      params: {},
    });

    tauri.invoke.mockResolvedValue({ ok: true, result: { pong: true } });
    await expect(backend.request("system.ping")).resolves.toEqual({ pong: true });
    expect(tauri.invoke).toHaveBeenLastCalledWith("plugin:cubesprite-mobile|request", {
      command: "system.ping",
      params: {},
    });
  });

  it("maps a native business error envelope to BackendRequestError", async () => {
    tauri.invoke
      .mockResolvedValueOnce({ ok: true, result: { protocol_version: 1 } })
      .mockResolvedValueOnce({
        ok: false,
        error: {
          code: "STALE_REVISION",
          message: "state changed",
          details: { expected: 2 },
        },
      });
    const backend = new TauriMobileBackend();
    await backend.start();

    await expect(backend.request("game.move", {})).rejects.toMatchObject({
      code: "STALE_REVISION",
      message: "state changed",
      details: { expected: 2 },
    } satisfies Partial<BackendRequestError>);
  });

  it("closes the native backend through the plugin", async () => {
    tauri.invoke.mockResolvedValue({ ok: true, result: {} });
    const backend = new TauriMobileBackend();
    await backend.start();
    await backend.close();

    expect(tauri.invoke).toHaveBeenCalledWith("plugin:cubesprite-mobile|close");
    await expect(backend.request("game.state")).rejects.toThrow(
      "CubeSprite mobile backend has not started.",
    );
  });
});
