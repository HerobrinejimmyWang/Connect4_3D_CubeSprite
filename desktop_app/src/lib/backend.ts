import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type { BackendApi, BackendErrorShape, InitializationResult } from "../types";

interface ResponseEnvelope {
  v: 1;
  type: "response";
  id: number | null;
  ok: boolean;
  result?: unknown;
  error?: BackendErrorShape;
}

interface EventEnvelope {
  v: 1;
  type: "event";
  event: string;
  data?: unknown;
}

interface PendingRequest {
  resolve: (result: unknown) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout> | null;
}

export const DEFAULT_REQUEST_TIMEOUT_MS = 15 * 60 * 1000;
export const REPLAY_ANALYSIS_TIMEOUT_MS = null;

export function requestTimeoutMs(command: string): number | null {
  return command === "replay.analyze"
    ? REPLAY_ANALYSIS_TIMEOUT_MS
    : DEFAULT_REQUEST_TIMEOUT_MS;
}

export class BackendRequestError extends Error {
  readonly code: string;
  readonly details: unknown;

  constructor(error: BackendErrorShape) {
    super(error.message);
    this.name = "BackendRequestError";
    this.code = error.code;
    this.details = error.details;
  }
}

function eventText(payload: unknown): string {
  if (typeof payload === "string") return payload;
  if (payload && typeof payload === "object") {
    const record = payload as Record<string, unknown>;
    for (const key of ["line", "data", "text", "message"]) {
      if (typeof record[key] === "string") return record[key] as string;
    }
    try {
      return JSON.stringify(payload);
    } catch {
      // Fall through to the generic conversion below.
    }
  }
  return String(payload ?? "");
}

export class TauriSidecarBackend implements BackendApi {
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private unlisten: UnlistenFn[] = [];
  private started = false;
  private buffer = "";
  private terminalError: Error | null = null;
  private lifecycleTail: Promise<void> = Promise.resolve();

  start(): Promise<InitializationResult> {
    return this.runLifecycle(() => this.startUnlocked());
  }

  private async startUnlocked(): Promise<InitializationResult> {
    if (!this.started) {
      this.terminalError = null;
      await this.attachListeners();
      try {
        await invoke("start_sidecar");
        if (this.terminalError) throw this.terminalError;
        this.started = true;
      } catch (error) {
        this.started = false;
        await this.detachListeners();
        throw error;
      }
    }
    return this.request<InitializationResult>("system.initialize");
  }

  async request<T>(command: string, params: Record<string, unknown> = {}): Promise<T> {
    if (!this.started) throw this.terminalError ?? new Error("CubeSprite backend has not started.");
    const id = this.nextId++;
    const line = JSON.stringify({ v: 1, type: "request", id, command, params });
    const result = new Promise<T>((resolve, reject) => {
      const timeoutMs = requestTimeoutMs(command);
      const timeout = timeoutMs === null
        ? null
        : setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`Backend request timed out: ${command}`));
        }, timeoutMs);
      this.pending.set(id, {
        resolve: resolve as (result: unknown) => void,
        reject,
        timeout,
      });
    });
    try {
      await invoke("write_sidecar", { line });
    } catch (error) {
      const pending = this.pending.get(id);
      if (pending) {
        if (pending.timeout !== null) clearTimeout(pending.timeout);
        this.pending.delete(id);
        pending.reject(this.terminalError ?? (error instanceof Error ? error : new Error(String(error))));
      }
    }
    return result;
  }

  close(): Promise<void> {
    return this.runLifecycle(() => this.closeUnlocked());
  }

  private async closeUnlocked(): Promise<void> {
    try {
      // A fatal startup event marks the protocol as stopped before Rust has
      // necessarily observed process termination. Always clear Rust's slot so
      // a retry cannot accidentally reuse a dying sidecar.
      await invoke("stop_sidecar");
    } catch (error) {
      console.error("Unable to stop CubeSprite backend cleanly", error);
    }
    this.started = false;
    this.rejectAll(new Error("CubeSprite backend stopped."));
    await this.detachListeners();
  }

  private runLifecycle<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.lifecycleTail.then(operation, operation);
    this.lifecycleTail = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  private async attachListeners(): Promise<void> {
    if (this.unlisten.length) return;
    this.unlisten.push(
      await listen<unknown>("sidecar-stdout", ({ payload }) => this.acceptOutput(eventText(payload))),
      await listen<unknown>("sidecar-stderr", ({ payload }) => {
        const line = eventText(payload).trim();
        if (line) console.error(`[CubeSprite backend] ${line}`);
      }),
      await listen<unknown>("sidecar-exit", ({ payload }) => {
        this.started = false;
        const error = this.terminalError ?? new Error(`CubeSprite backend exited (${eventText(payload)}).`);
        this.terminalError = error;
        this.rejectAll(error);
      }),
    );
  }

  private acceptOutput(chunk: string): void {
    this.buffer += chunk;
    if (!chunk.endsWith("\n")) this.buffer += "\n";
    const lines = this.buffer.split(/\r?\n/);
    this.buffer = lines.pop() ?? "";
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      let message: ResponseEnvelope | EventEnvelope | { type: string };
      try {
        message = JSON.parse(line) as ResponseEnvelope | EventEnvelope | { type: string };
      } catch {
        console.error(`[CubeSprite backend] Invalid JSONL output: ${line}`);
        continue;
      }
      if (message.type === "event") {
        const event = message as EventEnvelope;
        if (event.event === "backend.fatal") {
          const data = event.data && typeof event.data === "object" ? event.data as Record<string, unknown> : {};
          const error = new BackendRequestError({
            code: typeof data.code === "string" ? data.code : "BACKEND_FATAL",
            message: typeof data.message === "string" ? data.message : "CubeSprite backend failed to start.",
            details: data.details,
          });
          this.terminalError = error;
          this.started = false;
          this.rejectAll(error);
        }
        continue;
      }
      if (message.type !== "response") continue;
      const response = message as ResponseEnvelope;
      if (response.id === null) continue;
      const pending = this.pending.get(response.id);
      if (!pending) continue;
      if (pending.timeout !== null) clearTimeout(pending.timeout);
      this.pending.delete(response.id);
      if (response.ok) pending.resolve(response.result);
      else pending.reject(new BackendRequestError(response.error ?? { code: "UNKNOWN", message: "Unknown backend error" }));
    }
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) {
      if (pending.timeout !== null) clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.pending.clear();
  }

  private async detachListeners(): Promise<void> {
    for (const unlisten of this.unlisten.splice(0)) unlisten();
    this.buffer = "";
  }
}

export const sidecarBackend = new TauriSidecarBackend();
