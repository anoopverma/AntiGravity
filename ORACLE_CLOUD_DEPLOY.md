# 🚀 Hosting AntiGravity on Oracle Cloud (Step-by-Step)

This guide will help you deploy your AntiGravity dashboard to an Oracle Cloud (OCI) Compute Instance using Docker.

## 1. Create a Compute Instance
1. Log in to your [Oracle Cloud Console](https://cloud.oracle.com/).
2. Go to **Compute** -> **Instances** -> **Create Instance**.
3. **Image**: Choose `Ubuntu 22.04` (recommended) or `Oracle Linux 8`.
4. **Shape**: `VM.Standard.E4.Flex` (Free Tier eligible if available).
5. **Networking**: Ensure it has a public IP address.
6. **SSH Keys**: Download your Private Key (`.key`)—you'll need this to login.

## 2. Open Port 5002 (Web Access)
1. In the OCI Console, go to **Networking** -> **Virtual Cloud Networks**.
2. Select your VCN -> **Security Lists** -> **Default Security List**.
3. Click **Add Ingress Rules**:
   - **Source CIDR**: `0.0.0.0/0`
   - **IP Protocol**: `TCP`
   - **Destination Port Range**: `5002`
   - **Description**: `AntiGravity Dashboard`
4. On the VM itself, open the firewall:
   ```bash
   sudo ufw allow 5002/tcp
   ```

## 3. Install Docker on the VM
Once you SSH into your VM (`ssh -i your_key.key ubuntu@YOUR_IP`), run:
```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Logout and login back for group changes to take effect
```

## 4. Deploy the App
1. Clone your repository:
   ```bash
   git clone https://github.com/anoopverma/AntiGravity.git
   cd AntiGravity
   ```
2. Create your `.env` file on the server:
   ```bash
   nano .env
   ```
   *Paste your `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` here.*

3. Start everything:
   ```bash
   docker-compose up -d
   ```

## 5. Access Your Dashboard
Open your browser and go to:
`http://YOUR_VM_PUBLIC_IP:5002`

---
### 🛠 Maintenance
- **View Logs**: `docker-compose logs -f app`
- **Restart**: `docker-compose restart app`
- **Update Code**: `git pull && docker-compose build && docker-compose up -d`
