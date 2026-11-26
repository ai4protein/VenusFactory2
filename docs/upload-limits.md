# File uploads

`WEBUI_V2_MODE` determines the application upload limit:

| Mode | Single-file limit |
| --- | --- |
| `local` | No application size limit |
| `online` | 10 × 1024 × 1024 bytes (10 MB in the UI); the boundary is inclusive |

The shared `web_v2.upload_limits` policy covers chat attachments, Scientist inputs, Researcher files and pasted structures, model-plaza datasets, custom model training/prediction inputs, and legacy quick/advanced upload endpoints. The local Library uses chunked, atomic writes for uploads and replacements.

The browser validates file sizes before transfer. The server independently counts bytes and returns HTTP 413 for an oversized online upload; it never saves a truncated file. Failed replacements preserve the previous file.

Scientist accepts up to 32 files per batch on local deployments, with no application-level size limit. Scientist runs are available only in local mode.

Reverse-proxy body limits are separate deployment settings. A local reverse proxy must not impose a body-size ceiling if unrestricted local uploads are needed. For online multipart requests, the proxy must allow multiple valid files plus multipart overhead; a 10 MB total-request ceiling would incorrectly reject batches of smaller files.
