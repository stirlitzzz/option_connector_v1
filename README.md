1. FastAPI app
mkdir -p ~/deltadisco && cd ~/deltadisco
python3 -m venv .venv
. .venv/bin/activate
pip install fastapi uvicorn

cat > app.py << 'PY'
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health(): return {"ok": True}
PY


2. Systemd service for FastAPI

Keeps it running across reboots

sudo tee /etc/systemd/system/fastapi.service >/dev/null <<'UNIT'
[Unit]
Description=FastAPI (deltadisco)
After=network.target

[Service]
User=paperspace
WorkingDirectory=/home/paperspace/deltadisco
ExecStart=/home/paperspace/deltadisco/.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
UNIT

    sudo systemctl daemon-reload
    sudo systemctl enable --now fastapi


Test:/

    curl http://127.0.0.1:8000/health

3. Cloudflared (once per VM)

Install (binary method is the most reliable):

    wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb

Check:

    cloudflared --version

4. Login + tunnel

    cloudflared tunnel login          # copy/paste URL into your local browser
    cloudflared tunnel create deltadisco-api
    cloudflared tunnel route dns deltadisco-api deltadisco.party

5. Config + service

    sudo mkdir -p /etc/cloudflared
    sudo tee /etc/cloudflared/config.yml >/dev/null <<'YAML'
    tunnel: <TUNNEL-UUID-HERE>
    credentials-file: /etc/cloudflared/<TUNNEL-UUID-HERE>.json
    ingress:
      - hostname: deltadisco.party
        service: http://127.0.0.1:8000
      - service: http_status:404
    YAML

    sudo cp ~/.cloudflared/<TUNNEL-UUID-HERE>.json /etc/cloudflared/
    sudo chown root:root /etc/cloudflared/*.json

    sudo cloudflared service install
    sudo systemctl enable --now cloudflared


6. Verify

Open in browser:

    https://deltadisco.party/health
You should see: {"ok": true}
