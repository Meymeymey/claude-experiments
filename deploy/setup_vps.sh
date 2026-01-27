#!/bin/bash
# Deployment script for Hydrogen Simulation on Hostinger VPS
# Run as root: bash setup_vps.sh

set -e

echo "================================================"
echo "  Hydrogen Simulation - VPS Setup"
echo "================================================"

# Update system
echo "[1/7] Updating system..."
apt update && apt upgrade -y

# Install Python and dependencies
echo "[2/7] Installing Python and build tools..."
apt install -y python3 python3-pip python3-venv python3-dev build-essential
apt install -y nginx certbot python3-certbot-nginx
apt install -y git curl

# Create app user
echo "[3/7] Creating app user..."
useradd -m -s /bin/bash hydrogen || echo "User already exists"

# Create app directory
echo "[4/7] Setting up application directory..."
APP_DIR="/home/hydrogen/app"
mkdir -p $APP_DIR
chown -R hydrogen:hydrogen /home/hydrogen

# Copy application files (run this from your local machine or upload manually)
echo "[5/7] Application directory created at: $APP_DIR"
echo "Upload your files to this directory:"
echo "  - controller.py"
echo "  - simulation.py"
echo "  - network.py"
echo "  - controller.html"
echo "  - documentation.html"
echo "  - network_data.json"
echo "  - requirements.txt"

# Create virtual environment and install dependencies
echo "[6/7] Creating virtual environment..."
sudo -u hydrogen bash << 'EOF'
cd /home/hydrogen/app
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
EOF

# Install and configure systemd service
echo "[7/7] Installing systemd service..."
cat > /etc/systemd/system/hydrogen.service << 'EOF'
[Unit]
Description=Hydrogen Simulation Web Application
After=network.target

[Service]
Type=simple
User=hydrogen
Group=hydrogen
WorkingDirectory=/home/hydrogen/app
Environment="PATH=/home/hydrogen/app/venv/bin"
ExecStart=/home/hydrogen/app/venv/bin/gunicorn --workers 2 --bind 127.0.0.1:5000 controller:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable hydrogen
systemctl start hydrogen

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Upload your application files to: $APP_DIR"
echo "2. Run: systemctl restart hydrogen"
echo "3. Configure nginx (see nginx config file)"
echo "4. Set up SSL with: certbot --nginx -d yourdomain.com"
echo ""
echo "Check status with: systemctl status hydrogen"
echo "View logs with: journalctl -u hydrogen -f"
