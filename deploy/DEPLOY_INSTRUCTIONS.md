# Deployment Instructions for Hostinger VPS

## Your VPS Details
- **IP**: 72.62.35.98
- **OS**: Ubuntu 24.04
- **SSH**: `ssh root@72.62.35.98`

## Quick Deploy (3 Steps)

### Step 1: Initial Server Setup (run once)

SSH into your VPS and run the setup script:

```bash
ssh root@72.62.35.98

# Upload setup script first (from PowerShell on Windows):
# scp deploy/setup_vps.sh root@72.62.35.98:/root/

# Then on the VPS:
chmod +x /root/setup_vps.sh
bash /root/setup_vps.sh
```

### Step 2: Upload Application Files

From PowerShell on your Windows machine:

```powershell
cd c:\Users\hmk\claude\hydrogen_simulation

# Upload all files
scp controller.py simulation.py network.py controller.html documentation.html network_data.json requirements.txt root@72.62.35.98:/home/hydrogen/app/
```

Or use the deploy script:
```powershell
.\deploy\deploy.ps1
```

### Step 3: Install Dependencies & Start

SSH into your VPS:

```bash
ssh root@72.62.35.98

# Install Python packages
cd /home/hydrogen/app
sudo -u hydrogen bash -c 'source venv/bin/activate && pip install -r requirements.txt && pip install gunicorn highspy'

# Restart the service
systemctl restart hydrogen

# Check if it's running
systemctl status hydrogen
```

## Configure Nginx (for port 80)

```bash
# Copy nginx config
cp /root/nginx.conf /etc/nginx/sites-available/hydrogen

# Enable the site
ln -s /etc/nginx/sites-available/hydrogen /etc/nginx/sites-enabled/

# Remove default site (optional)
rm /etc/nginx/sites-enabled/default

# Test and reload
nginx -t
systemctl reload nginx
```

## Access Your App

- **Direct**: http://72.62.35.98
- **Phone**: Open the same URL in your mobile browser

## Optional: Add a Domain & SSL

If you have a domain (e.g., hydrogen.yourdomain.com):

1. Point your domain's DNS A record to 72.62.35.98
2. Update nginx.conf with your domain name
3. Run certbot for free SSL:

```bash
certbot --nginx -d hydrogen.yourdomain.com
```

## Useful Commands

```bash
# View application logs
journalctl -u hydrogen -f

# Restart application
systemctl restart hydrogen

# Check status
systemctl status hydrogen

# View nginx logs
tail -f /var/log/nginx/error.log

# Update code after changes
cd /home/hydrogen/app
sudo -u hydrogen git pull  # if using git
systemctl restart hydrogen
```

## Remove n8n (Optional)

If you want to free up resources by removing n8n:

```bash
# Stop n8n
docker stop n8n
docker rm n8n

# Or if using Docker Compose
cd /path/to/n8n
docker-compose down

# Remove Docker images (optional)
docker image prune -a
```

## Troubleshooting

### App won't start
```bash
# Check logs
journalctl -u hydrogen -n 50

# Test manually
cd /home/hydrogen/app
sudo -u hydrogen bash -c 'source venv/bin/activate && python controller.py'
```

### Solver not working
```bash
# Install HiGHS solver
sudo -u hydrogen bash -c 'source venv/bin/activate && pip install highspy'
```

### Port 5000 already in use
```bash
# Find what's using it
lsof -i :5000

# Kill if needed
kill -9 <PID>
```
