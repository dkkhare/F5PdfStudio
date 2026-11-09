# F5PdfStudio — LTX API worker pool

The Video Production tab has a shared worker pool for the LTX-2.3 image+audio
workflow and the LTX-2.5 pose-reference workflow.

## Register workers

1. Start F5PdfStudio and open **Video Production**.
2. Choose **LTX-2.3 / LTX-2.5 Pose — LTX API worker pool**.
3. Enter one worker's public **LTX API URL** and its **API Key**.
4. Confirm the two workflow JSON paths point to the LTX-2.3 and LTX-2.5 Pose
   API-format workflow files.
5. Click **Register LTX API worker**.
6. Repeat with the URL and key for each additional server, up to five.

Registration checks `GET /health`, then sends each workflow to
`POST /workflows/register` with `X-API-Key`. The returned workflow IDs,
URL and API key are saved in `data/ltx_api_workers.db`; they are not written
to individual project state files.

## Worker scheduling

- A worker is atomically reserved as **BUSY** before a job is submitted.
- The manager selects the least-recently-used **FREE** worker that has the
  workflow ID required by the selected LTX mode.
- LTX jobs use `POST /jobs`, poll `GET /jobs/{job_id}`, read
  `GET /jobs/{job_id}/outputs`, and download the selected output.
- The worker is released after the output is downloaded or the job fails.
- Up to five LTX generation callbacks can run concurrently. If all workers
  are busy, another generation waits for a worker to become free.
- Health checks run every 30 seconds and before a worker is selected.
  An unreachable server or a server reporting degraded ComfyUI health is
  marked **OFFLINE**. Three consecutive failed checks delete its database row,
  including its API key and registered workflow IDs.

The worker table shows status, current job, failed health checks, and both
workflow IDs. Use **Check all workers / Refresh** for an immediate health
check, or **Remove worker** to delete a registration manually.
