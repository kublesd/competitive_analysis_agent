[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [switch]$NoBrowser,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$applicationPath = Join-Path $projectRoot "competitive_analysis_agent\streamlit_app.py"
$environmentPath = Join-Path $projectRoot ".env"
$requiredVariables = @(
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "TAVILY_API_KEY"
)
$browserJob = $null

function Test-PythonCandidate {
    param(
        [string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        return $false
    }

    # 实际执行一次，避免把 Windows Store 的占位 python.exe 当成可用解释器。
    try {
        & $Executable @PrefixArguments -c "import sys; print(sys.executable)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Resolve-ProjectPython {
    if ($PythonPath) {
        $resolvedPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
            $PythonPath
        )
        if (-not (Test-PythonCandidate -Executable $resolvedPath)) {
            throw "The Python executable is unavailable: $resolvedPath"
        }
        return @{
            Executable = $resolvedPath
            PrefixArguments = @()
        }
    }

    $candidatePaths = @()
    if ($env:CONDA_PREFIX) {
        $candidatePaths += Join-Path $env:CONDA_PREFIX "python.exe"
    }
    $candidatePaths += Join-Path $projectRoot ".venv\Scripts\python.exe"
    $candidatePaths += Join-Path $env:USERPROFILE "miniconda3\python.exe"
    $candidatePaths += Join-Path $env:USERPROFILE "anaconda3\python.exe"

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source) {
        $candidatePaths += $pythonCommand.Source
    }

    foreach ($candidatePath in $candidatePaths) {
        if (Test-PythonCandidate -Executable $candidatePath) {
            return @{
                Executable = $candidatePath
                PrefixArguments = @()
            }
        }
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if (
        $pyCommand -and
        (Test-PythonCandidate -Executable $pyCommand.Source -PrefixArguments @("-3"))
    ) {
        return @{
            Executable = $pyCommand.Source
            PrefixArguments = @("-3")
        }
    }

    throw "No usable Python interpreter was found. Activate the Conda environment or pass -PythonPath."
}

function Test-ModelConfiguration {
    if (-not (Test-Path -LiteralPath $environmentPath -PathType Leaf)) {
        throw (
            "Configuration file is missing: $environmentPath. " +
            "Copy .env.example to .env and fill in your local API keys."
        )
    }

    $fileValues = @{}
    foreach ($rawLine in Get-Content -LiteralPath $environmentPath) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            continue
        }

        $parts = $line.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($requiredVariables -contains $name -and $value) {
            $fileValues[$name] = $true
        }
    }

    $missingVariables = @()
    foreach ($variableName in $requiredVariables) {
        $processValue = [Environment]::GetEnvironmentVariable($variableName)
        if (-not $processValue -and -not $fileValues.ContainsKey($variableName)) {
            $missingVariables += $variableName
        }
    }

    if ($missingVariables.Count -gt 0) {
        $missingText = $missingVariables -join ", "
        $helpMessages = @(
            " Fill .env with the missing values before starting."
        )
        $missingLlmVariables = @(
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL"
        ) | Where-Object { $missingVariables -contains $_ }

        if ($missingLlmVariables.Count -gt 0) {
            $helpMessages += (
                " Configure LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL " +
                "for your OpenAI-compatible model provider."
            )
        }
        if ($missingVariables -contains "TAVILY_API_KEY") {
            $helpMessages += (
                " Configure TAVILY_API_KEY for web search. " +
                "Get a key at https://app.tavily.com."
            )
        }
        $helpText = $helpMessages -join ""
        throw "Missing application configuration: $missingText.$helpText"
    }
}

$python = Resolve-ProjectPython
$pythonExecutable = $python.Executable
$pythonPrefixArguments = $python.PrefixArguments

Push-Location $projectRoot
try {
    # 在启动服务器前验证依赖，错误会直接显示在当前终端中。
    & $pythonExecutable @pythonPrefixArguments -c (
        "import streamlit, langchain_openai, competitive_analysis_agent"
    )
    if ($LASTEXITCODE -ne 0) {
        throw 'Project dependencies are missing. Run: python -m pip install -e ".[dev,llm]"'
    }

    Test-ModelConfiguration

    Write-Host "Project check passed."
    Write-Host "Python: $pythonExecutable"
    Write-Host "App URL: http://127.0.0.1:$Port"

    if ($CheckOnly) {
        return
    }

    Write-Host "Starting Streamlit. Press Ctrl+C to stop."
    $applicationUrl = "http://127.0.0.1:$Port"

    if (-not $NoBrowser) {
        # 等待健康端点成功后再打开浏览器，避免首次启动停在 Streamlit 邮箱提示。
        $browserJob = Start-Job -ScriptBlock {
            param([string]$Url)

            $healthUrl = "$Url/_stcore/health"
            $deadline = (Get-Date).AddSeconds(60)
            while ((Get-Date) -lt $deadline) {
                try {
                    $response = Invoke-WebRequest `
                        -Uri $healthUrl `
                        -UseBasicParsing `
                        -TimeoutSec 2
                    if ($response.StatusCode -eq 200) {
                        Start-Process $Url
                        return
                    }
                }
                catch {
                    Start-Sleep -Milliseconds 500
                }
            }
        } -ArgumentList $applicationUrl
    }

    # 始终使用 headless 服务模式，由上面的健康检查负责打开浏览器。
    & $pythonExecutable @pythonPrefixArguments -m streamlit run $applicationPath `
        --server.address 127.0.0.1 `
        --server.port $Port `
        --server.headless true `
        --browser.serverAddress 127.0.0.1 `
        --browser.serverPort $Port `
        --browser.gatherUsageStats false

    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE."
    }
}
finally {
    if ($null -ne $browserJob) {
        Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
