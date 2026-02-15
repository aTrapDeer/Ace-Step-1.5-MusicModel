param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8008,
    [switch]$Reload,
    [switch]$NoBrowser,
    [switch]$SkipNpmInstall,
    [switch]$SkipBuild
)

$cmd = @("python", "af3_gui_app.py", "--host", $BindHost, "--port", "$Port")
if ($Reload) { $cmd += "--reload" }
if ($NoBrowser) { $cmd += "--no-browser" }
if ($SkipNpmInstall) { $cmd += "--skip-npm-install" }
if ($SkipBuild) { $cmd += "--skip-build" }

$exe = $cmd[0]
$args = @()
if ($cmd.Length -gt 1) {
    $args = $cmd[1..($cmd.Length-1)]
}
& $exe @args
