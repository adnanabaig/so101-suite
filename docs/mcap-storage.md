# MCAP storage on the G15

Verified 2026-10-27: migration completed for all 1,365 inventoried recordings,
423,468,310,174 bytes (394.4 GiB). Destination receipts, file sizes and all
legacy symlinks passed audit; no regular local MCAP files or transfer partials
remained. The laptop recovered about 394 GiB free. A new managed recording on
SigmaDrive was closed and decoded with CRC validation. Recording now waits for
an explicit activity instead of running continuously. The private audit and capture reports are retained
under `.artifacts/`.

Raw recordings use SigmaDrive as their primary storage. The workstation's
ignored `config/rigs/mcap-storage.local.json` selects the output directory,
local metadata directory, expected mount and volume UUID, and a 50 GiB free
space reserve. The recording unit explicitly requires that profile.

On this rig, raw files belong under
`/media/adnan/SigmaDrive1/so101-suite/mcap`. Small `status.json` and
`episodes.json` files stay in `data/real/mcap` so the existing API can read
recorder health and logical episode boundaries without relocating its database.
Legacy raw paths are preserved as symlinks after migration.

## Missing or full drive

The recorder checks the mount, volume UUID, destination containment and free
space before creating output and on each recording pass. An absent/wrong/full
volume blocks recording; it never creates a replacement recording directory
on the laptop. The unit does not repeatedly restart for storage-blocked exit
78 or configuration exit 2. Existing recordings are retained.

Reconnect the same drive at its configured mount point, inspect storage and
then explicitly restart the recorder:

```bash
systemctl --user restart so101-recording.service
./services status
```

Use the workstation's `./services up` to start the full suite after a reboot.
Mount SigmaDrive first. Service availability does not arm either motor bus or
establish readiness for physical motion. Before unplugging the drive, stop the
recorder, confirm the current file closed and use the OS eject action.

## Migration evidence

The authorized migration keeps an immutable inventory, per-file SHA-256
receipts and a progress record in `so101-suite/mcap-migration` on SigmaDrive.
Each source is copied, flushed and read back for hash comparison before its
local bytes are removed. A symlink then preserves its original path. A stopped
migration resumes from the same manifest and receipts; it does not restart the
whole copy or discard previously verified progress.

Completion requires all manifest entries accounted for, destination sizes
matching receipts, legacy paths resolving correctly, and a fresh recorder file
created and decoded on the external drive. A progress file left at `moving`
does not prove a transfer process is still alive. Keep the recorder stopped
while the original inventory is being migrated.

## Capacity and retention

Measure actual bytes/hour for the selected camera modes, topic set and session
types. Divide free space above the reserve by that measured rate to estimate
remaining recording time. Five-minute segments simplify inspection but do not
reduce bytes/hour. More camera frames and raw depth increase storage demands.

The 2026-10-27 external-storage smoke produced 39.4 MB over 8.1 seconds with
the three existing image feeds and idle telemetry: approximately 16.3 GiB/hour.
This short sample is a capacity warning, not a sustained-rate guarantee.
This measures active capture only. The service creates no idle MCAPs; measure a
full selected episode before estimating the ten-episode storage budget.

Pin calibration evidence, accepted teaching episodes, evaluation trials and
failure examples. Keep a catalog linking episode IDs, segment hashes,
calibration versions and derived datasets. Mark idle diagnostic periods
separately. Automatic deletion is not configured; establish an explicit
retention policy before removing data.

Evaluate video compression on a copy with task-relevant image checks, decode
latency and seeking tests. MCAP supports chunk compression, but already
compressed JPEG frames may gain little from recompression. Preserve metric
depth losslessly for calibration/perception analysis. Never replace originals
until a versioned derived copy has been validated and retention is approved.
See the [MCAP specification](https://mcap.dev/spec) for compression formats.

SigmaDrive holds the primary copy after migration. It is not a backup of
itself. A second destination and selected data must be approved before any
external upload; uploads remain disabled.

## Isolated recorder verification

`scripts/record_mcap.py --output <directory> --state-dir <directory>` allows an
isolated short capture with `--duration`. An explicit output bypasses automatic
local-profile discovery; the managed unit always passes its required profile.
Do not use an override to bypass an unavailable production recording volume.
The capture helper observes existing local feeds and has no motor authority.

## Measured pilot and depth budget

The eight-second external-drive pilot on 2026-10-27 produced two completed,
CRC-checked segments totaling 39.6 MB with three RGB cameras and idle telemetry:
about 16.6 GiB/hour if that short sample were sustained. This is an initial
measurement, not a long-run capacity guarantee. For comparison, uncompressed
640×480 16-bit depth at 30 fps alone is 18.4 MB/s, about 61.8 GiB/hour, before
metadata or compression. Budget depth separately before continuous capture.

The service stays ready between episodes without creating MCAP files. It records
only a requested episode, a supported calibration/diagnostic operation, or an
explicit bounded diagnostic capture. Ending the activity closes its file;
stopping the service also closes the current file. Live camera previews and
Foxglove viewing can continue without disk capture. Storage guards and the free
space reserve still apply during every capture.
