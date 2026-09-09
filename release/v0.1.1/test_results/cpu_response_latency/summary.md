# Desktop CPU response-latency matrix

- Method: 22 deterministic non-terminal fixed states, independently reconstructed; returned actions are not applied.
- Temperature: 0.4; forced tactical shortcuts enabled; 10-second idle after each non-final response and 60 seconds between groups. Idle time is excluded.
- The v0.1.0 source replay is unavailable locally, so this corpus is deterministic but not directly position-for-position comparable to v0.1.0.
- A blank cell means a response exceeded 60 seconds and that group was stopped.

| Model \ MCTS simulations | 32 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| v3_b6c128 | 0.099s | 0.492s | 0.987s | 2.190s | 4.909s |
| v3_b8c192 | 0.140s | 0.612s | 1.228s | 2.741s | 5.979s |
| v3_b10c256 | 0.208s | 0.864s | 1.854s | 3.759s | 8.378s |
