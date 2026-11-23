<#
.SYNOPSIS
  Saves a screenshot of one window (for example the Unity Editor), or a region of it, as a PNG for a lesson's image
  section. Windows only; uses the window's visible frame, so no desktop shows around it.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools/capture-window.ps1 -Title "Unity 6" -Out assets/images/unity/editor-layout.png
  Captures the first window whose title contains "Unity 6" (the Unity Editor's title ends with the Unity version).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools/capture-window.ps1 -Title "Unity 6" -Crop 1180,90,420,640 -Out assets/images/unity/inspector.png
  Captures a region of that window: x, y, width and height in pixels, measured from the window's top-left corner.

.NOTES
  -List prints the processes with a window and the window size, without titles; -List -Titles adds the titles
  (they can contain private information, such as file names and browser tabs).
  Wider captures are scaled down to -MaxWidth pixels (default 1600) so lesson images stay small.
#>
param(
  [string]$Title,
  [string]$Out,
  [string]$Crop,                      # "x,y,width,height" inside the window
  [int]$MaxWidth = 1600,
  [switch]$List,
  [switch]$Titles
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class CaptureWin {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr h, int attr, out RECT r, int size);
  public static RECT Frame(IntPtr h) {
    RECT r;
    // DWMWA_EXTENDED_FRAME_BOUNDS (9): the visible frame, without the invisible resize borders of GetWindowRect
    if (DwmGetWindowAttribute(h, 9, out r, Marshal.SizeOf(typeof(RECT))) != 0) GetWindowRect(h, out r);
    return r;
  }
}
"@
[CaptureWin]::SetProcessDPIAware() | Out-Null     # real pixels on scaled (high-DPI) displays

$windows = Get-Process | Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero -and $_.MainWindowTitle }
if ($List) {
  foreach ($p in $windows) {
    $r = [CaptureWin]::Frame($p.MainWindowHandle)
    $line = '{0,-28} {1,5} x {2,-5}' -f $p.ProcessName, ($r.Right - $r.Left), ($r.Bottom - $r.Top)
    if ($Titles) { $line += '  ' + $p.MainWindowTitle }
    $line
  }
  exit 0
}
if (-not $Title -or -not $Out) { throw 'Give -Title (part of the window title) and -Out (a .png path), or -List.' }

$p = $windows | Where-Object { $_.MainWindowTitle -like "*$Title*" } | Select-Object -First 1
if (-not $p) { throw "No open window has '$Title' in its title. Run with -List -Titles to see the windows." }
$h = $p.MainWindowHandle
if ([CaptureWin]::IsIconic($h)) { [CaptureWin]::ShowWindow($h, 9) | Out-Null }   # SW_RESTORE a minimised window
[CaptureWin]::SetForegroundWindow($h) | Out-Null
Start-Sleep -Milliseconds 500                                                   # let it repaint in front

$r = [CaptureWin]::Frame($h)
$x = $r.Left; $y = $r.Top; $w = $r.Right - $r.Left; $hh = $r.Bottom - $r.Top
if ($Crop) {
  $c = @($Crop -split '[,\s]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
  if ($c.Count -ne 4) { throw '-Crop takes four numbers: x,y,width,height (relative to the window).' }
  $x += $c[0]; $y += $c[1]; $w = [Math]::Min($c[2], $w - $c[0]); $hh = [Math]::Min($c[3], $hh - $c[1])
}
if ($w -le 0 -or $hh -le 0) { throw "Nothing to capture: the region is $w x $hh pixels." }

$bmp = New-Object System.Drawing.Bitmap $w, $hh
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($x, $y, 0, 0, $bmp.Size)
$g.Dispose()
if ($w -gt $MaxWidth) {                                                        # scale down, keeping the shape
  $nw = $MaxWidth; $nh = [int][Math]::Round($hh * $MaxWidth / $w)
  $small = New-Object System.Drawing.Bitmap $nw, $nh
  $g2 = [System.Drawing.Graphics]::FromImage($small)
  $g2.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
  $g2.DrawImage($bmp, 0, 0, $nw, $nh)
  $g2.Dispose(); $bmp.Dispose(); $bmp = $small
}
$full = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path (Get-Location).Path $Out }
$full = [System.IO.Path]::GetFullPath($full)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $full) | Out-Null
$bmp.Save($full, [System.Drawing.Imaging.ImageFormat]::Png)
'{0}: {1} x {2} pixels, {3:N0} KB' -f $Out, $bmp.Width, $bmp.Height, ((Get-Item $full).Length / 1KB)
$bmp.Dispose()
