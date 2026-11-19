# Installing the suite

The first supported machine is an ASUS ROG Zephyrus G15 (RTX 3070 Laptop GPU,
8 GB) booted into Ubuntu 24.04 LTS (x86_64), with both arms and an Intel
RealSense camera plugged straight into it. `install/install.sh` is written for
that machine first. Its macOS path and `install/install.ps1` for Windows are
kept but unverified (see the end of this page).

For a laptop without the source tree, build a handoff folder on the
development machine (`make handoff OUT=<dir>`, below) and follow its
`START_HERE.md`.

## Ubuntu 24.04 (Zephyrus)

1. Install the NVIDIA driver: `sudo ubuntu-drivers install`. With Secure Boot
   on, choose "Enroll MOK" at the next boot and enter the password you set, or
   the driver doesn't load. Hybrid graphics is fine for CUDA. After a reboot
   `nvidia-smi` lists the GPU and the driver version.
2. From the suite folder (or a handoff folder's top level):

   ```bash
   install/install.sh --realsense        # handoff folder: bash install.sh --realsense
   install/install.sh --dry-run          # print every step, change nothing
   ```

   Run a live install as your normal user, not with `sudo`. It asks for sudo
   only for each system change that needs it. A root/container audit may use
   `--dry-run`; that mode changes nothing.

3. Log out and back in, so the `dialout` group and `~/.local/bin` take effect.
4. Run `so101 doctor` (from any folder) and fix every FAIL.

### What the installer does

| Step | Details |
| --- | --- |
| Python | The system `python3.12` (Ubuntu 24.04's), else `python3.13`, in a virtual environment at `~/.local/opt/so101-suite/venv`. It never downloads another Python. Python 3.14 and free-threaded builds (`3.13t`) are refused: torchcodec, PyAV, MuJoCo and OpenCV have no free-threaded wheels. |
| System packages | `python3.12-venv`, `ffmpeg`, `libusb-1.0-0`, `git`, `libegl1`, `libgl1`. The missing ones are printed as one `sudo apt-get update && sudo apt-get install ...` command, run only if you answer y. |
| PyTorch | torch 2.11.0 and torchvision 0.26.0 from one explicit index, then torchcodec 0.11.1 from an explicit PyPI resolve. The build follows the driver; see the table below. After the suite resolve, the installer reads installed metadata and refuses unless all three exact versions/build labels still match the selected profile. Pip and uv run with a private HOME/XDG tree, no user config, netrc or keyring, no inherited index setting and an environment allowlist that retains only package transport/CA settings. |
| The suite | The wheel (or the source folder) with the `robot,sim,train` extras, plus `realsense` with `--realsense`. Before changing the environment, the installer verifies that all dependency, extra, console-script and wheel metadata plus package files match the adjacent source and that its stamped web UI is current and intact. Selected extras must exist. A wheel is copied to a private read-only temporary directory before verification. A source install likewise validates and installs one private read-only snapshot. The selected snapshot is hash-checked before and after both pip uses, so replacing or editing the original USB/path cannot change the install and a snapshot mutation fails closed. feetech-servo-sdk has no wheel and builds locally; that is expected. |
| Commands | `so101` and `so101-suite` are linked into `~/.local/bin`, which Ubuntu puts on PATH at login. |
| Serial ports | If you aren't in `dialout`, it prints `sudo usermod -aG dialout $USER` and runs it only if you answer y. |
| Device ownership | Creates `/var/lib/so101/device-ownership-v2` and its lock/journal subdirectories as root:`dialout`, mode 3770 (setgid+sticky). This stable, shared namespace is independent of `TEMP`, checkout and data folder; real-arm commands fail closed if it is missing or unsafe. macOS uses the matching root:`staff` tree under `/Library/Application Support/SO101`; Windows uses per-user `%LOCALAPPDATA%\SO101\device-ownership-v2`. |
| Arm names | With `--follower-serial` and `--leader-serial`, udev rules in `/etc/udev/rules.d/99-so101-arms.rules` name the adapters `/dev/so101-follower` and `/dev/so101-leader` and keep ModemManager off them (`ENV{ID_MM_DEVICE_IGNORE}="1"`). Shown first, applied only if you answer y. |
| Settings file | `~/.local/share/so101/.env` gets the absolute `SO101_DATA_DIR`, `SO101_DATASETS_DIR` and `SO101_OUTPUTS_DIR`, and with the serial numbers `SO101_FOLLOWER_PORT=/dev/so101-follower` and `SO101_LEADER_PORT=/dev/so101-leader`. It's your file, so no sudo; each added line is shown. A setting already there with another value is kept unless you answer y. The install fails if its data folder cannot be created and written. |
| RealSense | With `--realsense`, Intel's udev rules go to `/etc/udev/rules.d/99-realsense-libusb.rules`, shown first and applied only if you answer y. An existing suite or system rules file counts only when valid rules cover all five target models: D405, D415, D435, D435i and D455. A truncated or stale file is offered for replacement. |
| Doctor | `python -m so101.platform_info` and `so101 doctor` run at the end. |
| Launcher | "SO-101 Suite" in the app menu (`~/.local/share/applications/so101-suite.desktop`): it starts the suite in a terminal and opens the browser. The generated Linux and macOS launchers pin the resolved data folder with `--home`, so a custom `SO101_HOME` still applies when the desktop session does not inherit the shell environment. `--no-launcher` skips it. |

The installer never deletes the data folder or removes a setting, and never runs
sudo in a dry run or without a terminal: it prints the commands instead.

### PyTorch builds

| NVIDIA driver (`nvidia-smi`) | Profile | Where torch comes from |
| --- | --- | --- |
| 580 or newer | `cu130` | PyPI (built for CUDA 13.0) |
| 570 to 579 | `cu128` | `https://download.pytorch.org/whl/cu128` |
| 525 to 569 | `cu126` | `https://download.pytorch.org/whl/cu126` |
| none, or older than 525 | `cpu` | `https://download.pytorch.org/whl/cpu` |

`--profile cuda` picks from the driver and fails without one; `--profile cpu`,
`cu130`, `cu128` or `cu126` force a build. Below the driver NVIDIA lists for
the build's CUDA update (580.95.05, 570.124.06 and 560.35.05) the installer
warns that the GPU runs in CUDA minor-version compatibility mode. The RTX
3070 (sm_86) is in every CUDA build.

If `nvidia-smi` is installed but doesn't answer, the NVIDIA kernel module
isn't loaded: finish the MOK enrollment, or switch a laptop in
integrated-only mode to hybrid, then re-run with `--profile cuda`.

### Stable names for the arms

`/dev/ttyACM0` and `/dev/ttyACM1` can swap when you replug. Plug both arms in,
read each adapter's serial number from `python -m so101.platform_info`, then:

```bash
install/install.sh --follower-serial 585A0077531 --leader-serial 585A0077541
```

Unplug and replug both arms. The installer also sets the ports in the data
folder's `.env` (these are the suite's defaults on Linux anyway):

```
SO101_FOLLOWER_PORT=/dev/so101-follower
SO101_LEADER_PORT=/dev/so101-leader
```

If that file already names other ports, it asks before replacing them.

If a CH340/CH341 adapter never gets a serial port, `brltty` has claimed it
(doctor warns about this): `sudo apt remove brltty`, then replug.

### Simulation rendering

With the NVIDIA driver loaded, MuJoCo renders offscreen on the GPU with
`MUJOCO_GL=egl`. The launcher sets it; set it yourself
(`export MUJOCO_GL=egl`) when you run `so101` by hand. Doctor renders one frame
with the setting the suite uses and warns if that fails. Without a GPU, use
`MUJOCO_GL=osmesa` (`sudo apt install libosmesa6`).

### The data folder

Settings (`.env`), recordings, datasets and checkpoints live in
`~/.local/share/so101` (`$XDG_DATA_HOME/so101`; `SO101_HOME` overrides it).
`python -m so101.platform_info --data-dir` prints the path.

Every `so101` command and the launcher read their settings the same way,
whatever folder they run in. Highest priority first:

1. `SO101_*` variables in the environment;
2. `.env` in the current folder (a source checkout's own);
3. `.env` in the data folder;
4. the built-in defaults.

The installer (and the launcher, when the file doesn't exist yet) writes
absolute `SO101_DATA_DIR`, `SO101_DATASETS_DIR` and `SO101_OUTPUTS_DIR` into
the data folder's `.env`, so data lands there from any folder. Without them the
three folders are `./data`, `./datasets` and `./outputs` in the current folder,
which is what a source checkout uses. `so101 info` and `so101 doctor` print the
settings files and folders in effect; doctor warns when the data folder's
`.env` leaves the storage folders to the current folder.

A source checkout on a machine with the suite installed reads the installed
`.env` too, under its own `.env`. To keep the checkout apart, set the storage
folders (and anything else) in the checkout's `.env`, or run with
`SO101_HOME=$PWD`.

The workstation install refuses a missing, stale or altered web UI. For an
intentional API/CLI-only development environment, opt out explicitly with both
`--no-launcher --allow-no-ui`; `--allow-no-ui` alone is refused so the installer
cannot create a browser launcher for a package that has no operator UI.
Paths containing pip requirement delimiters (`;`, `[`, `]`, `#` or `?`) are
refused before installation; move the source/handoff to a plain path.

### Keep the laptop awake

A sleeping or suspended laptop can't stop the arm: the servos keep holding
their last goal. While the arm is powered, keep the lid open and turn off
automatic suspend (Settings > Power > Automatic Suspend), or:

```bash
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
```

(`sleep-inactive-battery-type` on battery.) Doctor checks the AC setting on a
laptop.

### Doctor

`so101 doctor` prints the suite's checks, then the platform section: Python,
the GPU and the CUDA build torch sees, `SO101_POLICY_DEVICE`, MuJoCo's GL
setting and a test render, serial ports and their permissions, the arm and
RealSense udev rules, RealSense cameras and their USB link (USB 2 limits
resolution and frame rate), free disk space, brltty, and laptop sleep.
`so101 doctor --json > doctor.json` writes all of it as JSON. Its `backend`
field names the backend that ran the checks (`real`, `mock` or `sim`; under
`sim` the `sim` field holds the `SO101_SIM_*` settings), and the table ends
with the same line. Only a `real` report says anything about the arm: sim and
mock runs share the data folder with it but are rehearsals. Under `sim`,
doctor checks that MuJoCo and the simulated devices load and skips the serial
port and calibration checks; `so101 info` shows the sim settings instead of
the ports.

## The handoff folder

On the development machine, from a clean commit:

```bash
make handoff OUT=/media/usb/so101-handoff
```

For the promoted G15 candidate, publish one downloadable archive only from the
full commit that was approved by the software gate:

```bash
archive="$PWD/../so101-handoff-aaaaaaaaaaaa.tar.gz"
make handoff-archive \
  HANDOFF_APPROVED_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  OUT="$archive"
(
  cd "$(dirname "$archive")"
  sha256sum --check "$(basename "$archive").sha256"
)
```

The archive target checks `HEAD` against the approved commit before and after
construction and refuses any drift. It never overwrites either the archive or
its external `.sha256` sidecar; if publishing the sidecar fails, it rolls back
the archive so the same output can be retried. The archive has one deterministic
top-level folder, `so101-handoff/`. Extract it, enter that folder, then verify
the inner manifest before running anything from the bundle:

```bash
archive="$PWD/../so101-handoff-aaaaaaaaaaaa.tar.gz"
(
  cd "$(dirname "$archive")"
  tar -xzf "$(basename "$archive")"
  cd so101-handoff
  sha256sum --check SHA256SUMS
)
```

Using `bash` is intentional and continues to work when copying or extracting
through a filesystem that drops executable bits. Follow the generated
`START_HERE.md` from the fresh extraction: it runs the M0 software gate from
inside `so101/` before any install. Only after M0 passes does it run
`bash install.sh --dry-run`. The bundled clone sets clone-local
`core.filemode=false`, so mode-only loss does not make M0 call it dirty; Git
still checks tracked content and untracked files, and the inner SHA-256 manifest
still checks the copied bytes.

It holds a clone of the current branch with its history (`so101/`), the wheel
with the web UI built in (`wheels/`, so the laptop needs no Node.js), a
top-level `install.sh` that installs that wheel, `START_HERE.md` and
`SHA256SUMS` (`sha256sum -c SHA256SUMS` checks the copy). The builder refuses
uncommitted changes and untracked files, and never writes into the repository.
It builds beside the requested output and publishes the folder only when
complete, so a failed build can be retried with the same output path. A
populated output folder is never removed or overwritten.

The wheel is built from the clone's exact commit (a detached checkout of it),
so a commit landing on the branch meanwhile can't change it. With
`--wheel FILE` (`python scripts/make_handoff.py --out DIR --wheel FILE`) it
uses a wheel you built, and checks it the same way: its METADATA version is
the commit's, its `so101/` files are byte for byte the commit's `src/so101`,
and its UI's build stamp names this commit and the commit's frontend sources.
An intentionally UI-less handoff (`--allow-no-ui`) generates a backend-only
wrapper that always forwards `--no-launcher --allow-no-ui`; it cannot silently
create a workstation launcher without an operator UI.

Start the installer with `bash install.sh` on the laptop. A FAT32 or exFAT USB
stick (udisks mounts it with `showexec`) drops the exec bit from every file,
so `./install.sh` fails there with "Permission denied"; the wrapper runs the
real installer through `bash` as well.

## Behind a TLS-inspecting proxy

Export `PIP_CERT=<CA bundle>` for pip (and `NODE_EXTRA_CA_CERTS` for npm when
building a handoff folder) before running the installer.

## macOS and Windows (unverified)

These paths are kept as they are and have not been run on their targets.

- macOS (Apple Silicon): `install/install.sh` installs the PyPI torch build,
  which uses the Apple GPU (MPS). uv, if installed, provides Python 3.12.
  With `--realsense` before macOS 15, only Python 3.12 works
  (pyrealsense2-macosx 2.54.2). librealsense usually needs sudo on macOS;
  color-only capture also works through the camera's UVC index.
- Windows 10/11 (x86-64): `powershell -ExecutionPolicy Bypass -File
  install\install.ps1` (`-Profile`, `-RealSense`, `-DryRun`). It hasn't been
  run or linted (no PowerShell was available). AMD Radeon GPUs, such as the
  RX 5700 XT, have no supported PyTorch GPU path, so torch uses the CPU.
  Long paths: if an install fails on a long path, turn them on in an
  administrator PowerShell with `New-ItemProperty -Path
  HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled
  -Value 1 -PropertyType DWORD -Force`.
