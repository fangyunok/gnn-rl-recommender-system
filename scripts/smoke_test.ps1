param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$UserId = 1
)

$health = Invoke-RestMethod "$BaseUrl/health"
if ($health.status -ne "ok") { throw "Health check failed" }
$recommendations = Invoke-RestMethod "$BaseUrl/v1/recommendations/$UserId`?limit=5"
if ($recommendations.items.Count -ne 5) { throw "Recommendation response is incomplete" }

Write-Output "Health check passed for dataset=$($health.dataset)"
Write-Output ($recommendations | ConvertTo-Json -Depth 5)

