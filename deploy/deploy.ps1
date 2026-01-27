# PowerShell script to deploy to Hostinger VPS
# Run from: c:\Users\hmk\claude\hydrogen_simulation\

$VPS_IP = "72.62.35.98"
$VPS_USER = "root"
$APP_DIR = "/home/hydrogen/app"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Deploying Hydrogen Simulation to VPS" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Files to upload
$files = @(
    "controller.py",
    "simulation.py",
    "network.py",
    "controller.html",
    "documentation.html",
    "network_data.json",
    "requirements.txt"
)

Write-Host "`n[1/3] Uploading application files..." -ForegroundColor Yellow

foreach ($file in $files) {
    Write-Host "  Uploading $file..."
    scp $file "${VPS_USER}@${VPS_IP}:${APP_DIR}/"
}

Write-Host "`n[2/3] Uploading deployment scripts..." -ForegroundColor Yellow
scp deploy/setup_vps.sh "${VPS_USER}@${VPS_IP}:/root/"
scp deploy/nginx.conf "${VPS_USER}@${VPS_IP}:/root/"

Write-Host "`n[3/3] Setting permissions and restarting service..." -ForegroundColor Yellow
ssh "${VPS_USER}@${VPS_IP}" "chown -R hydrogen:hydrogen ${APP_DIR} && systemctl restart hydrogen"

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "  Deployment Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "`nAccess your app at: http://${VPS_IP}" -ForegroundColor Cyan
Write-Host "Check status: ssh ${VPS_USER}@${VPS_IP} 'systemctl status hydrogen'" -ForegroundColor Gray
