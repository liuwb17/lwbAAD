param([switch]$Continuous)
# Read-only physical-core load probe for Windows / WSL callers (9950X3D, <=64 logical CPUs).
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public static class AadCoreTopology {
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool GetLogicalProcessorInformation(IntPtr buffer, ref uint length);
    public static int[][] Read() {
        if (IntPtr.Size != 8 || Environment.ProcessorCount > 64)
            throw new Exception("Expected a 64-bit host with <=64 logical CPUs");
        uint length = 0;
        GetLogicalProcessorInformation(IntPtr.Zero, ref length);
        if (length == 0 || length % 32 != 0) throw new Exception("Invalid CPU topology size");
        IntPtr buffer = Marshal.AllocHGlobal((int)length);
        try {
            if (!GetLogicalProcessorInformation(buffer, ref length))
                throw new Exception("Cannot read physical-core topology");
            var groups = new List<int[]>();
            for (int offset = 0; offset < length; offset += 32) {
                IntPtr row = IntPtr.Add(buffer, offset);
                if (Marshal.ReadInt32(row, 8) != 0) continue; // RelationProcessorCore
                ulong mask = unchecked((ulong)Marshal.ReadInt64(row));
                var cpus = new List<int>();
                for (int cpu = 0; cpu < 64; cpu++) if ((mask & (1UL << cpu)) != 0) cpus.Add(cpu);
                groups.Add(cpus.ToArray());
            }
            return groups.ToArray();
        } finally { Marshal.FreeHGlobal(buffer); }
    }
}
'@
$groups = [AadCoreTopology]::Read()
# Resolve localized counter names by their English registry indices.
$english = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Perflib\009').Counter
$localized = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Perflib\CurrentLanguage').Counter
$names = @{}
for ($i = 0; $i -lt $localized.Length; $i += 2) { $names[$localized[$i]] = $localized[$i + 1] }
$processorName = $null
$timeName = $null
for ($i = 0; $i -lt $english.Length; $i += 2) {
    if ($english[$i + 1] -eq 'Processor') { $processorName = $names[$english[$i]] }
    if ($english[$i + 1] -eq '% Processor Time') { $timeName = $names[$english[$i]] }
}
if (!$processorName -or !$timeName) { throw 'CPU performance counter names unavailable' }
function Write-CoreSample($sample) {
$loads = @{}
foreach ($counter in $sample.CounterSamples) {
    if ($counter.InstanceName -match '^\d+$') {
        if ($counter.Status -notin @(0, 1)) { throw 'Invalid CPU performance counter sample' }
        $loads[[int]$counter.InstanceName] = [Math]::Max(0, [Math]::Min(100, $counter.CookedValue))
    }
}
$coreLoads = @()
foreach ($group in $groups) {
    $busy = 0.0
    foreach ($cpu in $group) {
        if (!$loads.ContainsKey($cpu)) { throw 'Missing logical CPU sample' }
        $busy = [Math]::Max($busy, $loads[$cpu])
    }
    $coreLoads += $busy
}
@{ core_loads = $coreLoads; sample_seconds = 1 } | ConvertTo-Json -Compress

}
if ($Continuous) {
    Get-Counter -Counter "\$processorName(*)\$timeName" -SampleInterval 1 -Continuous | ForEach-Object { Write-CoreSample $_ }
} else {
    $samples = @(Get-Counter -Counter "\$processorName(*)\$timeName" -SampleInterval 1 -MaxSamples 2)
    Write-CoreSample $samples[-1]
}
