<#
.SYNOPSIS
Install or upgrade the SO-101 suite on Windows (Linux and macOS: install.sh).

.DESCRIPTION
Creates a virtual environment under -Prefix, installs the PyTorch build for
this machine, then the suite with its extras, runs "so101 doctor", and adds
SO-101 Suite to the Start Menu and the desktop. Re-run it to upgrade.

It never deletes or rewrites the data folder (%LOCALAPPDATA%\so101: datasets,
checkpoints, .env). On the first run doctor creates its empty subfolders and
audit database there, as the first launch would.

UNVERIFIED: this script was written without a Windows machine or PowerShell
to run it on. It targets Windows PowerShell 5.1 and PowerShell 7.

Run it from the suite folder:

    powershell -ExecutionPolicy Bypass -File install\install.ps1
    powershell -ExecutionPolicy Bypass -File install\install.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File install\install.ps1 -Profile cpu
    powershell -ExecutionPolicy Bypass -File install\install.ps1 -RealSense

Profiles (the only difference is which PyTorch build is installed):

    cuda   NVIDIA GPU. Picks cu128 or cu126 from the driver version.
    cu128  torch 2.11.0+cu128 from download.pytorch.org
    cu126  torch 2.11.0+cu126 from download.pytorch.org
    cpu    torch 2.11.0+cpu from download.pytorch.org (no NVIDIA GPU; AMD
           Radeon cards such as the RX 5700 XT train on the CPU)

-RealSense adds the suite's realsense extra (pyrealsense2) for Intel RealSense
cameras. Its Windows wheels (2.56) stop at Python 3.13, so with -RealSense the
installer refuses a newer Python.

Needs: Windows 10 or 11 (64-bit x86), uv or Python 3.12 from python.org (not
the Microsoft Store), internet access. Behind a TLS-inspecting proxy, set
UV_SYSTEM_CERTS=1 (uv) or PIP_CERT=<CA bundle> (pip) first.

.PARAMETER TorchProfile
auto (default), cuda, cu128, cu126 or cpu. Also accepted as -Profile.

.PARAMETER Prefix
Where the app goes. Default: %LOCALAPPDATA%\Programs\so101-suite

.PARAMETER Python
A Python 3.12 python.exe to build the environment from (3.12 or 3.13 with
-RealSense). Default: uv if it is installed, else "py -3.12".

.PARAMETER Extras
Suite extras. Default: robot,sim,train. An empty string installs the suite
alone (mock mode and the web UI).

.PARAMETER Wheel
Install this wheel. Default: the newest dist\so101-<version>-*.whl of the source
folder's version (pyproject.toml), refused if src\so101 or frontend\src changed
after it was built; else the source folder. The wheel's METADATA must have the
version its file name claims.

.PARAMETER RealSense
Add Intel RealSense camera support: the same as adding realsense to -Extras.

.PARAMETER Fresh
Delete and recreate the app's virtual environment (not the data).

.PARAMETER NoLauncher
Don't add the Start Menu and desktop shortcuts.

.PARAMETER DryRun
Print the commands instead of running them.
#>
[CmdletBinding()]
param(
    [Alias('Profile')]
    [ValidateSet('auto', 'cuda', 'cu128', 'cu126', 'cpu')]
    [string]$TorchProfile = 'auto',
    [string]$Prefix = '',
    [string]$Python = '',
    [string]$Extras = 'robot,sim,train',
    [string]$Wheel = '',
    [switch]$RealSense,
    [switch]$Fresh,
    [switch]$NoLauncher,
    [switch]$DryRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# Versions LeRobot 0.6.1 accepts (torch>=2.7,<2.12; torchvision>=0.22,<0.27).
$TorchVersion = '2.11.0'
$TorchvisionVersion = '0.26.0'
$TorchIndex = 'https://download.pytorch.org/whl'
# NVIDIA drivers for the CUDA builds (Windows numbers). The torch 2.11 wheels pin
# CUDA 12.8.1 and 12.6.3 (their metadata on download.pytorch.org), and NVIDIA's
# CUDA release notes (Table 4) list these drivers for those updates. Older R525+
# drivers still run CUDA 12 builds under minor-version compatibility (Table 3),
# which only rules out PTX JIT; the wheels ship native code for every GPU the
# suite targets, so they don't need it.
$MinDriverCu128 = [version]'572.61'
$MinDriverCu126 = [version]'561.17'
$MinDriverCuda12 = [version]'525.0'
$LaunchArgs = '-P -X utf8 -m so101.launcher --pause-on-error'

$Root = Split-Path -Parent $PSScriptRoot

function Say([string]$Message) { Write-Host "==> $Message" }
function Warn([string]$Message) { Write-Warning $Message }
function Fail([string]$Message) {
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

function Format-Arg([string]$Value) {
    if ($Value -eq '' -or $Value -match '[\s"]') { return '"' + $Value + '"' }
    return $Value
}

# Print a command; run it unless this is a dry run. Stops on a non-zero exit.
function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    $shown = @($Exe) + $Arguments | ForEach-Object { Format-Arg $_ }
    Write-Host ('+ ' + ($shown -join ' '))
    if ($DryRun) { return }
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "$Exe exited with code $LASTEXITCODE" }
}

# Run a command and return its output as UTF-8 text (Python is started with
# -X utf8 so paths with non-ASCII user names survive the pipe).
function Get-Utf8Output {
    param([string]$Exe, [string[]]$Arguments, [switch]$Quiet)
    $previousEncoding = [Console]::OutputEncoding
    try {
        try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
        if ($Quiet) { $output = & $Exe @Arguments 2>$null } else { $output = & $Exe @Arguments }
        return (($output | Out-String).Trim())
    } finally {
        try { [Console]::OutputEncoding = $previousEncoding } catch { }
    }
}

# Output of a probe command, or $null if it is missing or fails. Windows
# PowerShell 5.1 turns redirected stderr into errors, so relax the preference.
function Get-ProbeOutput {
    param([string]$Exe, [string[]]$Arguments)
    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) { return $null }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = Get-Utf8Output $Exe $Arguments -Quiet
        if ($LASTEXITCODE -ne 0) { return $null }
        return $output
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $previous
    }
}

# sys.executable, version and base prefix of a Python, or $null.
function Get-PythonInfo {
    param([string]$Exe, [string[]]$PreArgs = @())
    # No quote characters: Windows PowerShell 5.1 mangles them in native arguments.
    $code = 'import sys; print(sys.executable); print(*sys.version_info[:2], sep=chr(46)); print(sys.base_prefix)'
    $output = Get-ProbeOutput $Exe ($PreArgs + @('-X', 'utf8', '-c', $code))
    if (-not $output) { return $null }
    $lines = @($output -split "`r?`n")
    if ($lines.Count -lt 3) { return $null }
    try { $version = [version]$lines[1].Trim() } catch { return $null }
    return [pscustomobject]@{
        Executable = $lines[0].Trim()
        Version    = $version
        BasePrefix = $lines[2].Trim()
    }
}

function Get-NvidiaDriver {
    $output = Get-ProbeOutput 'nvidia-smi' @('--query-gpu=driver_version', '--format=csv,noheader')
    if (-not $output) { return $null }
    try { return [version](@($output -split "`r?`n")[0].Trim()) } catch { return $null }
}

function Get-CudaProfile($Driver) {
    if ($null -eq $Driver) { return $null }
    if ($Driver -ge $MinDriverCu128) { return 'cu128' }
    if ($Driver -ge $MinDriverCuda12) { return 'cu126' }
    return $null
}

function Show-OldDriverWarning([string]$Name, $Driver) {
    if ($null -eq $Driver) { return }
    $need = $null
    if ($Name -eq 'cu128') { $need = $MinDriverCu128 }
    if ($Name -eq 'cu126') { $need = $MinDriverCu126 }
    if ($null -eq $need -or $Driver -ge $need) { return }
    if ($Driver -ge $MinDriverCuda12) {
        Warn "NVIDIA driver $Driver is older than $need, which NVIDIA lists for the $Name build. It runs in CUDA minor-version compatibility mode; if training reports CUDA errors, update the driver (NVIDIA App or nvidia.com) and re-run."
    } else {
        Warn "NVIDIA driver $Driver is too old for CUDA 12 builds (they need $MinDriverCuda12 or newer); torch will use the CPU until you update it."
    }
}

# --- the machine ----------------------------------------------------------------

if ($env:OS -ne 'Windows_NT') { Fail 'this script is for Windows; on Linux or macOS run install/install.sh' }
$arch = $env:PROCESSOR_ARCHITECTURE
if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
if ($arch -ne 'AMD64') { Fail "this needs 64-bit x86 Windows (found $arch): MuJoCo has no Windows on Arm build" }

# -RealSense and realsense in -Extras mean the same thing.
$extrasList = @(($Extras -replace '\s', '') -split ',' | Where-Object { $_ })
$useRealSense = $RealSense.IsPresent -or ($extrasList -contains 'realsense')
if ($useRealSense -and -not ($extrasList -contains 'realsense')) { $extrasList += 'realsense' }
$Extras = $extrasList -join ','

# The Python versions this install can use: 3.12 or 3.13, the suite's
# requires-python (>=3.12,<3.14). The pyrealsense2 2.56 Windows wheels stop at 3.13 too.
$PythonMin = [version]'3.12'
$PythonMax = [version]'3.13'
$PythonWant = 'Python 3.12 or 3.13'
$PythonWhy = ''
function Test-PythonVersion($Version) {
    if ($null -eq $Version -or $Version -lt $PythonMin) { return $false }
    return ($null -eq $PythonMax -or $Version -le $PythonMax)
}

$gpuNames = @()
try { $gpuNames = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) } catch { }
$hasNvidiaSmi = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)

$profileName = $TorchProfile
if ($profileName -eq 'auto') {
    $driver = Get-NvidiaDriver
    $cuda = Get-CudaProfile $driver
    if ($cuda) {
        Say "NVIDIA driver ${driver}: using the $cuda PyTorch build"
        Show-OldDriverWarning $cuda $driver
        $profileName = $cuda
    } elseif ($driver) {
        Warn "NVIDIA driver $driver is too old for any CUDA 12 build (they need $MinDriverCuda12 or newer). Installing the CPU build; update the driver (NVIDIA App or nvidia.com) and re-run with -Profile cuda to use the GPU."
        $profileName = 'cpu'
    } else {
        if ($hasNvidiaSmi) {
            Warn 'nvidia-smi is installed but not answering. On an ASUS laptop in Eco mode the NVIDIA GPU is switched off: pick Standard or Optimized in Armoury Crate or G-Helper, then re-run with -Profile cuda. Installing the CPU build.'
        }
        $profileName = 'cpu'
    }
} elseif ($profileName -eq 'cuda') {
    $driver = Get-NvidiaDriver
    if (-not $driver) { Fail '-Profile cuda: nvidia-smi does not answer. Install the NVIDIA driver, or leave Eco mode on an ASUS laptop (or pass -Profile cu128 to install anyway).' }
    $profileName = Get-CudaProfile $driver
    if (-not $profileName) { Fail "NVIDIA driver $driver is too old: CUDA 12 builds need $MinDriverCuda12 or newer ($MinDriverCu128 for cu128)" }
    Say "NVIDIA driver ${driver}: using the $profileName PyTorch build"
    Show-OldDriverWarning $profileName $driver
} elseif ($profileName -ne 'cpu') {
    Show-OldDriverWarning $profileName (Get-NvidiaDriver)
}
if ($profileName -eq 'cpu' -and ($gpuNames -match 'AMD|Radeon') -and -not ($gpuNames -match 'NVIDIA')) {
    Say 'AMD Radeon GPU: PyTorch has no Windows build that uses it, so training runs on the CPU. The suite itself (control, recording, simulation) does not need a GPU.'
}

$torchSpecs = @("torch==$TorchVersion+$profileName", "torchvision==$TorchvisionVersion+$profileName")
$torchArgs = @('--index-url', "$TorchIndex/$profileName")

# --- where things go -----------------------------------------------------------------

if (-not $Prefix) {
    if (-not $env:LOCALAPPDATA) { Fail 'LOCALAPPDATA is not set; pass -Prefix' }
    $Prefix = Join-Path $env:LOCALAPPDATA 'Programs\so101-suite'
}
$Prefix = [System.IO.Path]::GetFullPath($Prefix)
$Venv = Join-Path $Prefix 'venv'
$Vpy = Join-Path $Venv 'Scripts\python.exe'

if (-not $Wheel) {
    # Only a wheel of the version this source folder builds, and not one older
    # than its sources: an old wheel left in dist\ would install old code.
    $sourceVersion = ''
    $pyproject = Join-Path $Root 'pyproject.toml'
    if (Test-Path -LiteralPath $pyproject) {
        $found = Select-String -LiteralPath $pyproject -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
        if ($found) { $sourceVersion = $found.Matches[0].Groups[1].Value }
    }
    $filter = if ($sourceVersion) { "so101-$sourceVersion-*.whl" } else { 'so101-*.whl' }
    $newest = Get-ChildItem -LiteralPath (Join-Path $Root 'dist') -Filter $filter -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest) {
        $changed = Get-ChildItem -LiteralPath (Join-Path $Root 'src\so101'), (Join-Path $Root 'frontend\src') -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notlike '*\__pycache__\*' -and $_.Extension -ne '.pyc' -and $_.LastWriteTime -gt $newest.LastWriteTime } |
            Select-Object -First 1
        if ($changed) { Fail "$($newest.FullName) is older than the source folder ($($changed.FullName) changed after it was built). Rebuild it with the web UI: make frontend-deps build-ui wheel. Or pass the wheel you mean with -Wheel." }
        $Wheel = $newest.FullName
    }
}
if ($Wheel) {
    if (-not (Test-Path -LiteralPath $Wheel -PathType Leaf)) { Fail "no such wheel: $Wheel" }
    $Target = [System.IO.Path]::GetFullPath($Wheel)
    $sourceDesc = "wheel $Target"
    # Its METADATA must have the version its file name claims.
    if ((Split-Path -Leaf $Target) -match '^so101-([^-]+)-.+\.whl$') { $claimed = $Matches[1] } else { Fail "$Target isn't named like a suite wheel (so101-<version>-py3-none-any.whl)" }
    $metaVersion = ''
    $readError = ''
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [System.IO.Compression.ZipFile]::OpenRead($Target)
        try {
            $entry = $zip.Entries | Where-Object { $_.FullName -match '^[^/]+\.dist-info/METADATA$' } | Select-Object -First 1
            if ($entry) {
                $reader = New-Object System.IO.StreamReader($entry.Open())
                try { $metadata = $reader.ReadToEnd() } finally { $reader.Dispose() }
                if ($metadata -match '(?m)^Version:\s*(\S+)') { $metaVersion = $Matches[1] }
            }
        } finally { $zip.Dispose() }
    } catch { $readError = $_.Exception.Message }
    if ($readError) { Fail "can't install ${Target}: it is not a readable wheel ($readError)" }
    if ($metaVersion -ne $claimed) { Fail "$Target is named version $claimed, but its METADATA says $(if ($metaVersion) { $metaVersion } else { 'nothing' }). Rebuild it (make frontend-deps build-ui wheel) or pass the right file with -Wheel." }
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $Root 'pyproject.toml'))) { Fail "no dist\so101-*.whl and no source folder at $Root" }
    $Target = $Root
    $sourceDesc = "source folder $Root"
    if (-not (Test-Path -LiteralPath (Join-Path $Root 'frontend\dist\index.html'))) {
        Warn 'frontend\dist is not built, so this install has no web UI (the API still works). Build it with Node.js (in the frontend folder: npm ci, then npm run build) and re-run, or pass a wheel built with "make frontend-deps build-ui wheel" as -Wheel.'
    }
}
if ($Extras) { $suiteSpec = $Target + '[' + $Extras + ']' } else { $suiteSpec = $Target }

$useUv = (-not $Python) -and [bool](Get-Command uv -ErrorAction SilentlyContinue)

Say "machine:  Windows $arch$(if ($gpuNames) { ' (' + ($gpuNames -join ', ') + ')' })"
Say "profile:  $profileName ($($torchSpecs -join ' '))"
Say "app:      $Prefix"
Say "suite:    $sourceDesc [$Extras]"
if ($useRealSense) { Say "realsense: yes (needs $PythonWant)" }
if ($useUv) { Say "using:    uv ($((Get-Command uv | Select-Object -First 1).Path))" }
if ($DryRun) { Say 'dry run: nothing will be changed' }

# Windows can't replace files a running suite has open.
$running = 0
foreach ($process in @(Get-Process -Name python, pythonw, so101, so101-suite -ErrorAction SilentlyContinue)) {
    try { $exePath = $process.Path } catch { $exePath = $null }
    if ($exePath -and $exePath.StartsWith($Venv, [System.StringComparison]::OrdinalIgnoreCase)) { $running++ }
}
if ($running -gt 0) { Fail 'SO-101 Suite is running from this install. Stop it (Ctrl+C in its window) and run the installer again.' }

try {
    $longPaths = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled
} catch { $longPaths = 0 }
if ($longPaths -ne 1) {
    Warn 'Windows long paths are off. If the install fails on a long file path, turn them on in an administrator PowerShell: New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force'
}

$needGb = 3
if ($profileName -like 'cu*') { $needGb = 8 }
try {
    $drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($Prefix))
    $freeGb = [math]::Floor($drive.AvailableFreeSpace / 1GB)
    if ($freeGb -lt $needGb) { Warn "only $freeGb GB free on $($drive.Name); the $profileName install may need about $needGb GB" }
} catch { }

# --- virtual environment -----------------------------------------------------------------

# The version of the venv's Python, or $null if it is missing or broken.
function Get-VenvVersion {
    if (-not (Test-Path -LiteralPath $Vpy)) { return $null }
    $info = Get-PythonInfo $Vpy
    if ($null -eq $info) { return $null }
    return $info.Version
}

function Remove-Venv {
    Write-Host "+ Remove-Item -Recurse -Force $(Format-Arg $Venv)"
    if ($DryRun) { return }
    try { Remove-Item -LiteralPath $Venv -Recurse -Force } catch { Fail "couldn't remove ${Venv}: $($_.Exception.Message). Close every SO-101 Suite window and try again." }
}

if ((Test-Path -LiteralPath $Venv) -and -not (Test-Path -LiteralPath (Join-Path $Venv 'pyvenv.cfg'))) {
    Fail "$Venv exists but is not a virtual environment; move it away or pick another -Prefix"
}
$venvVersion = Get-VenvVersion
$venvOk = Test-PythonVersion $venvVersion
if ($Fresh -and (Test-Path -LiteralPath $Venv)) {
    Say 'removing the old virtual environment (-Fresh)'
    Remove-Venv
    $venvOk = $false
} elseif ($null -ne $venvVersion -and $venvVersion -ge $PythonMin -and -not $venvOk) {
    Fail "$Venv has Python $venvVersion, but this install needs $PythonWant$PythonWhy. Re-run with -Fresh to recreate the app's virtual environment; the data folder is kept."
} elseif ((Test-Path -LiteralPath $Venv) -and -not $venvOk) {
    Say "the virtual environment at $Venv is broken or too old; recreating it"
    Remove-Venv
}

if ($venvOk) {
    Say "reusing $Venv (Python $venvVersion)"
} else {
    if (-not $DryRun) { New-Item -ItemType Directory -Path $Prefix -Force | Out-Null }
    if ($useUv) {
        # uv's own CPython build, never the Microsoft Store one.
        Invoke-Step 'uv' @('venv', '--seed', '--managed-python', '--python', '3.12', $Venv)
    } else {
        $info = $null
        if ($Python) {
            $info = Get-PythonInfo $Python
            if (-not $info) { Fail "can't run $Python" }
        } else {
            foreach ($candidate in @(@('py', '-3.12'), @('py', '-3.13'), @('python'))) {
                $preArgs = @()
                if ($candidate.Count -gt 1) { $preArgs = @($candidate[1..($candidate.Count - 1)]) }
                $found = Get-PythonInfo $candidate[0] $preArgs
                if ($found -and (Test-PythonVersion $found.Version)) { $info = $found; break }
            }
            if (-not $info) { Fail "$PythonWant not found$PythonWhy. Install uv (https://docs.astral.sh/uv/) or Python 3.12 from python.org, then run the installer again." }
        }
        if (-not (Test-PythonVersion $info.Version)) { Fail "$($info.Executable) is Python $($info.Version); this install needs $PythonWant$PythonWhy" }
        if ($info.BasePrefix -match '\\WindowsApps\\') {
            Fail "$($info.Executable) is the Microsoft Store Python. Its writes to AppData go to a private copy (docs.python.org/3.12/using/windows.html), which hides the suite's files. Install uv or Python 3.12 from python.org instead."
        }
        if ($info.Version -ne [version]'3.12') { Warn "using Python $($info.Version); the suite is tested on 3.12" }
        Invoke-Step $info.Executable @('-m', 'venv', $Venv)
    }
}

function Install-Packages([string[]]$Arguments) {
    if ($useUv) {
        Invoke-Step 'uv' (@('pip', 'install', '--python', $Vpy) + $Arguments)
    } else {
        Invoke-Step $Vpy (@('-m', 'pip', 'install', '--disable-pip-version-check') + $Arguments)
    }
}

# --- packages ------------------------------------------------------------------------------

if (-not $useUv) {
    Invoke-Step $Vpy @('-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', 'pip')
}
Say "step 1/3: PyTorch ($profileName)"
Install-Packages ($torchArgs + $torchSpecs)
Say 'step 2/3: the suite and its dependencies'
# torch is already satisfied, so this keeps the build from step 1. torchcodec
# comes from PyPI: the PyTorch CUDA indexes have no Windows torchcodec 0.11.
Install-Packages @($suiteSpec)
Say 'step 3/3: refresh the suite itself (same version numbers between builds)'
if ($useUv) {
    Install-Packages @('--no-deps', '--reinstall-package', 'so101', $Target)
} else {
    Install-Packages @('--no-deps', '--force-reinstall', $Target)
}
if ($useRealSense -and -not $DryRun) {
    # No quote characters: Windows PowerShell 5.1 mangles them in native arguments.
    $code = 'import importlib.util, sys; print(importlib.util.find_spec(sys.argv[1]) is not None)'
    if ((Get-ProbeOutput $Vpy @('-P', '-c', $code, 'pyrealsense2')) -ne 'True') {
        Warn "-RealSense: pyrealsense2 didn't get installed. This build of the suite may have no realsense extra (pip and uv only warn about that); see the install output above."
    }
}

# --- check -------------------------------------------------------------------------------------

$so101Exe = Join-Path $Venv 'Scripts\so101.exe'
if (-not $env:LOCALAPPDATA) { Fail 'LOCALAPPDATA is required for stable motor-device locks' }
$DeviceState = Join-Path $env:LOCALAPPDATA 'SO101\device-ownership-v2'
Say "motor-device ownership state: $DeviceState (independent of TEMP and the data folder)"
if ($DryRun) {
    Write-Host "+ New-Item -ItemType Directory -Force $DeviceState\locks, $DeviceState\intent-events\aliases"
} else {
    foreach ($folder in @(
        $DeviceState,
        (Join-Path $DeviceState 'locks'),
        (Join-Path $DeviceState 'intent-events'),
        (Join-Path $DeviceState 'intent-events\aliases')
    )) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
    }
}
if ($DryRun) {
    $DataDir = '(the data folder: so101.platform_info --data-dir)'
    Say "data folder: $DataDir"
    Invoke-Step $Vpy @('-m', 'so101.platform_info')
    Invoke-Step $so101Exe @('doctor')
} else {
    $DataDir = Get-Utf8Output $Vpy @('-X', 'utf8', '-m', 'so101.platform_info', '--data-dir')
    if ($LASTEXITCODE -ne 0 -or -not $DataDir) { Fail 'could not work out the data folder (so101.platform_info --data-dir)' }
    Say "data folder: $DataDir"
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    Push-Location -LiteralPath $DataDir
    try {
        Say 'platform report'
        & $Vpy -m so101.platform_info
        if ($LASTEXITCODE -ne 0) { Warn 'the platform report failed' }
        Say 'so101 doctor'
        & $so101Exe doctor
        if ($LASTEXITCODE -ne 0) { Fail 'so101 doctor failed to run; see the error above' }
    } finally {
        Pop-Location
    }
}

# --- launchers ----------------------------------------------------------------------------------

function New-Shortcut([string]$Path) {
    Write-Host "+ shortcut $(Format-Arg $Path) -> $(Format-Arg $Vpy) $LaunchArgs"
    if ($DryRun) { return }
    $folder = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($Path)
    $link.TargetPath = $Vpy
    $link.Arguments = $LaunchArgs
    $link.WorkingDirectory = $DataDir
    $link.Description = 'Operate, record, simulate and train the SO-101 arm'
    $link.Save()
}

if (-not $NoLauncher) {
    New-Shortcut (Join-Path ([Environment]::GetFolderPath('Programs')) 'SO-101 Suite.lnk')
    New-Shortcut (Join-Path ([Environment]::GetFolderPath('Desktop')) 'SO-101 Suite.lnk')
    Say 'launcher: SO-101 Suite in the Start Menu and on the desktop'
}

Say "done. Start SO-101 Suite from the Start Menu, or run: $(Join-Path $Venv 'Scripts\so101-suite.exe')"
Say "the arm's USB adapters show up as COM ports (Device Manager > Ports). Put them in $DataDir\.env, for example: SO101_FOLLOWER_PORT=COM3"
if ($useRealSense) { Say 'RealSense: plug the camera into a USB 3 port; USB 2 limits resolution and frame rate' }
