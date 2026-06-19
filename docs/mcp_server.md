# MCP Server

SceneForge exposes its pipeline as MCP tools, so an agent can drive
image→3D-scene generation as part of a larger workflow instead of only
through the CLI.

## Why async / job-based, not a single blocking call

The pipeline takes anywhere from minutes (COLMAP alone) to hours (CPU
OpenSplat training) to complete. A tool that blocks until the whole thing
finishes would hang the calling agent's turn for that entire duration with
no progress visibility. Instead:

- `start_pipeline` launches the run in a background thread and returns
  almost immediately with a `job_id`.
- `check_job_status` polls progress (`pending` / `running` / `completed` /
  `failed` / `aborted`).
- `get_viewer_path` retrieves the final output location once the job
  reaches `completed`.

This mirrors the poll-based job pattern used by cloud 3DGS APIs (Luma AI's
capture/slug model was the original reference point during this project's
design phase).

## Tools

| Tool | Purpose |
|---|---|
| `start_pipeline(images_dir, output_dir, iterations, use_gpu, force, skip_quality_gate)` | Start a run, returns `{job_id, status}` immediately |
| `check_job_status(job_id)` | Poll status, returns elapsed time, warnings, error if any |
| `get_viewer_path(job_id)` | Get `viewer.html` / `splat.ply` paths once completed |
| `list_jobs()` | List all jobs known to this server process |
| `check_environment()` | Check if COLMAP/OpenSplat/GPU are available before starting a run |

## Running the server

```bash
pip install -e .
sceneforge-mcp
```

This starts the server on stdio transport. To use it from Claude Desktop
or another MCP client, add it to your client's MCP server config, e.g.:

```json
{
  "mcpServers": {
    "sceneforge": {
      "command": "sceneforge-mcp"
    }
  }
}
```

## What was actually tested

- All 5 tools register correctly with proper schemas (verified via
  `mcp.list_tools()`).
- `check_environment()` correctly reflects real system state (COLMAP
  present, OpenSplat absent, no GPU -- the genuine state of the dev
  sandbox this was built in).
- A full async job lifecycle was run against REAL COLMAP execution:
  `start_pipeline` returns in <2 seconds while COLMAP's matching+mapping
  run for real in a background thread (~70+ seconds), and
  `check_job_status` correctly tracks the transition from `running` to
  `failed` once the (genuinely absent) OpenSplat stage is reached.
- The `PipelineAborted` quality-gate exception correctly maps to job
  status `aborted`, distinct from genuine failures.

## What was NOT tested

- The server has not been connected to a real MCP client (Claude Desktop,
  Claude Code, etc.) -- only the underlying tool functions and schema
  registration were tested directly in Python. The stdio transport layer
  itself (`mcp.run()`) is unexercised.
- Concurrent job handling under real load (the in-memory job registry uses
  a simple lock; it hasn't been stress-tested with many simultaneous jobs).
- Server restart behavior -- jobs are stored in-memory only and will be
  lost if the server process restarts mid-run.
