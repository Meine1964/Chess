$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8

function Find-Element($root, $property, $value) {
    $condition = [System.Windows.Automation.PropertyCondition]::new($property, $value)
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
}

function Get-Address($root) {
    $address = Find-Element $root ([System.Windows.Automation.AutomationElement]::AutomationIdProperty) 'view_1012'
    if ($null -eq $address) { throw 'Browser address bar unavailable.' }
    $pattern = $address.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
    return $pattern.Current.Value
}

function Test-Address($address, $gameId) {
    if ($address -notmatch '^https?://') { $address = 'https://' + $address }
    $uri = $null
    if (-not [Uri]::TryCreate($address, [UriKind]::Absolute, [ref]$uri)) { return $false }
    return $uri.Scheme -eq 'https' -and $uri.Host -eq 'lichess.org' -and
        ($uri.AbsolutePath -eq "/$gameId" -or $uri.AbsolutePath -eq "/game/$gameId")
}

function Find-Clocks($textPieces) {
    $clocks = @()
    foreach ($colon in @($textPieces | Where-Object { $_.name -eq ':' -and $_.height -ge 25 })) {
        $center = $colon.y + $colon.height / 2
        $line = @($textPieces | Where-Object {
            $_.height -ge $colon.height * 0.45 -and
            [Math]::Abs(($_.y + $_.height / 2) - $center) -le $colon.height * 0.5
        } | Sort-Object x, y, name -Unique)
        $group = @()
        $groups = @()
        $right = [double]::NegativeInfinity
        foreach ($piece in $line) {
            if ($group.Count -gt 0 -and $piece.x - $right -gt $colon.height * 0.5) {
                $groups += ,@($group)
                $group = @()
            }
            $group += $piece
            $right = [Math]::Max($right, $piece.x + $piece.width)
        }
        if ($group.Count -gt 0) { $groups += ,@($group) }
        foreach ($group in $groups) {
            if (@($group | Where-Object {
                $_.name -eq ':' -and $_.x -eq $colon.x -and $_.y -eq $colon.y
            }).Count -eq 0) { continue }
            $value = -join @($group | ForEach-Object { $_.name })
            if ($value -match '^(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2}(\.\d{1,2})?)$') {
                $clocks += [pscustomobject]@{
                    x = $group[0].x; y = $center; height = $colon.height; value = $value
                }
            }
        }
    }
    $clocks = @($clocks | Sort-Object x, y, value -Unique)
    if ($clocks.Count -ne 2) { throw 'Could not uniquely identify both browser clocks.' }
    $clocks = @($clocks | Sort-Object y)
    $clockHeight = [Math]::Max($clocks[0].height, $clocks[1].height)
    if ($clocks[1].y - $clocks[0].y -lt 2 * $clockHeight -or
        [Math]::Abs($clocks[1].x - $clocks[0].x) -gt 2 * $clockHeight) {
        throw 'Browser clock positions are ambiguous.'
    }
    return @{ top = $clocks[0].value; bottom = $clocks[1].value }
}

function Read-Clocks($request) {
    $root = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$request.hwnd)
    if ($null -eq $root -or $root.Current.IsOffscreen) { throw 'Browser window unavailable.' }
    if (-not (Test-Address (Get-Address $root) $request.game_id)) {
        throw 'Browser does not display the selected game.'
    }
    $element = [System.Windows.Automation.AutomationElement]
    $main = Find-Element $root $element::AutomationIdProperty 'main-wrap'
    if ($null -eq $main) { throw 'Lichess game content unavailable.' }
    $textCondition = [System.Windows.Automation.PropertyCondition]::new(
        $element::ControlTypeProperty, [System.Windows.Automation.ControlType]::Text
    )
    $ranks = @()
    $pieces = @()
    foreach ($text in $main.FindAll([System.Windows.Automation.TreeScope]::Descendants, $textCondition)) {
        $name = $text.Current.Name
        if ($name -match '^[0-9:.]+$' -and -not $text.Current.IsOffscreen) {
            $bounds = $text.Current.BoundingRectangle
            $pieces += [pscustomobject]@{
                x = $bounds.X; y = $bounds.Y; width = $bounds.Width
                height = $bounds.Height; name = $name
            }
            if ($name -match '^[1-8]$') {
                $ranks += [pscustomobject]@{ x = $bounds.X; y = $bounds.Y; value = [int]$name }
            }
        }
    }
    $ranks = @($ranks | Sort-Object x, y, value -Unique)
    foreach ($one in @($ranks | Where-Object value -eq 1)) {
        foreach ($eight in @($ranks | Where-Object value -eq 8)) {
            $step = ($eight.y - $one.y) / 7
            if ([Math]::Abs($one.x - $eight.x) -gt 3 -or [Math]::Abs($step) -lt 10) { continue }
            $matches = 0
            foreach ($number in 2..7) {
                $expectedY = $one.y + ($number - 1) * $step
                if (@($ranks | Where-Object {
                    $_.value -eq $number -and [Math]::Abs($_.x - $one.x) -le 3 -and
                    [Math]::Abs($_.y - $expectedY) -le 3
                }).Count -gt 0) { $matches++ }
            }
            if ($matches -eq 6) {
                $topColor = if ($step -lt 0) { 'black' } else { 'white' }
                if ($topColor -eq $request.player_color) {
                    throw 'Browser board is flipped; opponent clock is not on top.'
                }
            }
        }
    }
    $clocks = Find-Clocks $pieces
    if (-not (Test-Address (Get-Address $root) $request.game_id)) {
        throw 'Browser changed during clock reading.'
    }
    return $clocks
}

while ($null -ne ($line = [Console]::In.ReadLine())) {
    try {
        $request = $line | ConvertFrom-Json
        $response = Read-Clocks $request
    } catch {
        $response = @{ error = $_.Exception.Message }
    }
    [Console]::Out.WriteLine(($response | ConvertTo-Json -Compress))
}
