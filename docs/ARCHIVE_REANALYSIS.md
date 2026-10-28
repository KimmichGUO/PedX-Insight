# Private archive reanalysis

`scripts/reanalyze_archive.py` consumes the prepared private inventory queue. It accepts
only ready rows with a unique bare YouTube ID, a safe underscore-free existing name,
and explicit finite clip bounds. Blocked and ambiguous rows remain blocked. Work stays
under the external `--work-root` in media, results, state, and logs directories; old
repository results and tracked mapping/summary data are preserved.

Install downloads separately from CV. The official [yt-dlp EJS setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS)
requires its matching solver scripts: pip's `default` dependency group includes them.
Node 22 or newer must be on PATH; the worker enables it with `--js-runtimes node`.
FFmpeg and FFprobe must also be available. This download profile is separate from the
scientific CPU lock and contains no Torch/OpenCV dependencies.

```powershell
$DownloadEnv = Join-Path $env:LOCALAPPDATA 'PedX\environments\download-2026-10-28'
py -3.12 -m venv $DownloadEnv
& "$DownloadEnv\Scripts\python.exe" -m pip install -r requirements-download.txt
& "$DownloadEnv\Scripts\python.exe" -m pip check
```

The isolated Windows environment tested on October 28, 2026 uses yt-dlp 2026.10.21,
yt-dlp-ejs 0.8.0, and Node 24.18.0. The requirements file pins the direct download
profile, rather than claiming an artifact-hashed transitive lock. Keep its installed
freeze and resolver report with private run diagnostics.

Use explicit interpreters, and acquire a bounded pilot before inference:

```powershell
py -3.12 -B scripts/reanalyze_archive.py `
  --queue "$env:LOCALAPPDATA\PedX\reanalysis\2026-10-28\queue.csv" `
  --work-root 'E:\PedX-Reanalysis\2026-10-28' `
  --python 'PATH\TO\CV\python.exe' `
  --download-python "$DownloadEnv\Scripts\python.exe" `
  --video-id ULl8s4qydrk --clip-seconds 300 --prepare-only `
  --download-timeout-seconds 600
```

`--download-python` defaults to `--python` for compatibility. Use the isolated profile
when the CV interpreter lacks EJS. Cookies are copied from the existing local jar into
the acquisition directory and the temporary copy is removed on exit. Cookie values
are never stored in state. Raw download metadata and logs stay private.

The worker configures 30-second socket/network I/O timeouts in yt-dlp and FFmpeg, with three bounded retries
for extractor, HTTP, fragment, and file-access failures. The overall acquisition
deadline defaults to 3,600 seconds and terminates only its owned downloader/FFmpeg
process tree. `--download-timeout-seconds` must be finite and positive. Analysis and
localization have no deadline imposed by this option. The [FFmpeg protocol documentation](https://ffmpeg.org/ffmpeg-protocols.html)
defines `rw_timeout` in microseconds; the worker passes 30,000,000.

A normal public New York probe for seconds 18–21 loaded the isolated EJS package
without solver warnings, but media transfer still timed out at its 180-second
deadline. That probe verifies bounded failure handling; it does not establish source
availability. Keep source-access failures blocked rather than treating setup as proof
of a successful acquisition.

Acquisition requires matching video identity and section metadata, usable FFprobe
video/duration, and media/info hashes. Resuming reuses media only when those hashes and
requested bounds match. A pilot's scope and original historical interval remain
explicit, and pilot/full runs use different roots. Interrupted or failed rows retain
their state and media; other eligible rows can continue. An OS batch lock prevents two
workers from writing the same work root.

To analyze, omit `--prepare-only` and provide `--osm-python PATH\TO\OSM\python.exe`.
`--device gpu` waits for at least 8 GiB free VRAM and utilization at most 25%; it does
not stop other GPU jobs. `--device cpu` clears CUDA visibility for each child. A 20 GiB
free-disk guard runs before work. Results remain unverified pending a separate,
source-bound filming-location review. The worker never changes eligibility or publishes.
