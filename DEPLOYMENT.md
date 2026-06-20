# Eyewitness — ML Engineer Deployment Guide

**Production ML system for AI-powered accident analysis**  
*Computer Vision + Physics Simulation + LLM Integration + Observability*

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Production Deployment](#production-deployment)
5. [Model Monitoring & Observability](#model-monitoring--observability)
6. [API Integration](#api-integration)
7. [Troubleshooting](#troubleshooting)

---

## System Architecture

### Three-Layer Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  VIDEO INPUT (dashcam footage)                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: CV Facts (YOLO11 + OpenCV)                       │
│  • Vehicle detection, tracking, trajectory analysis         │
│  • Speed estimation, TTC calculation                        │
│  • Deterministic outputs: no hallucination risk            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: Avoidability Physics                             │
│  • Field of Safe Motion calculation                         │
│  • Reaction distance (1.5s human delay)                     │
│  • Braking physics (7 m/s² deceleration)                   │
│  • Verdict: Could collision have been avoided?             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: VLM Corroboration (Claude Sonnet 4-6)           │
│  • Multimodal context (4 keyframes + CV facts)            │
│  • Structured output (Pydantic validation)                 │
│  • Retry logic (tenacity: 3 attempts, exp backoff)        │
│  • Cost tracking ($0.003/1K input, $0.015/1K output)      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  EVIDENCE TRAIL (Butterbase append-only ledger)            │
│  • claims → facts → frames → fault_analyses → overrides    │
│  • Immutable audit log: never delete, always append        │
│  • Human-in-the-loop: override without data loss          │
└─────────────────────────────────────────────────────────────┘
```

### MLOps Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **CV Models** | YOLO11x (130 MB) | Vehicle detection & tracking |
| **Physics Engine** | NumPy kinematics | Avoidability calculation |
| **LLM** | Claude Sonnet 4-6 | Fault hypothesis generation |
| **Retry Logic** | Tenacity | Exponential backoff (1s → 10s) |
| **Validation** | Pydantic v2 | Structured output parsing |
| **Database** | Butterbase (Postgres) | Append-only evidence ledger |
| **UI** | Gradio 4.x | Interactive demo interface |
| **Monitoring** | Built-in cost + latency tracking | Per-request metrics |

---

## Prerequisites

### System Requirements

- **Python:** 3.11+ (tested on 3.11.x)
- **RAM:** 4 GB minimum (YOLO11x needs ~2 GB)
- **Disk:** 500 MB (YOLO weights auto-download)
- **Network:** Stable connection for LLM API calls

### API Keys

1. **Anthropic Claude** ([console.anthropic.com](https://console.anthropic.com))
   - Create API key with **Message Batches** permission
   - Costs: ~$0.001 per analysis (4 frames × 1024px)
   - Rate limit: 50 requests/min (Tier 1)

2. **Butterbase** ([butterbase.ai/dashboard](https://butterbase.ai/dashboard))
   - Create new app → copy `app_id` and API key
   - Free tier: 100k rows/month
   - Used for: Evidence storage, audit trail, override logging

---

## Local Development Setup

### Step 1: Clone & Install Dependencies

```bash
cd /Users/harshbajaj/Code/eyewitness

# Install Python packages (pip or conda)
pip install -r requirements.txt

# Or with venv isolation
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies breakdown:**
```
ultralytics       # YOLO11 inference
opencv-python     # Video processing, tracking
anthropic         # Claude API client
numpy             # Physics calculations
gradio>=4.0       # Web UI
Pillow            # Image handling
tenacity          # Retry logic with exp backoff
pydantic>=2.0     # Structured output validation
python-dotenv     # Secrets management
```

### Step 2: Configure Secrets (CRITICAL — Never commit `.env`)

Create `.env` file in project root:

```bash
cat > .env << 'EOF'
# ── Eyewitness — API Keys ──────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE
BUTTERBASE_API_KEY=bb_sk_YOUR_KEY_HERE

# ── Optional Overrides ─────────────────────────────────────
# YOLO_MODEL=yolo11n.pt          # Faster CPU inference (45 MB)
# CLAUDE_MODEL=claude-sonnet-4-6 # Or claude-3-5-sonnet-latest
# GRADIO_PORT=7860
# GRADIO_SHARE=false             # Set true for public URL
# USE_BUTTERBASE_GATEWAY=false   # Route LLM through Butterbase
EOF
```

**Security best practices:**
- ✅ Add `.env` to `.gitignore`
- ✅ Never hardcode keys in source files
- ✅ Rotate keys every 90 days
- ✅ Use separate keys for dev/staging/prod

### Step 3: Health Check (Pre-Flight Validation)

```bash
python scripts/health_check.py
```

**Expected output:**
```
======================================================
  EYEWITNESS  ·  Health Check
  https://api.butterbase.ai/v1/app_46yxrt8czo59
======================================================

──────────────────────────────────────────────────────
  1 · Environment
──────────────────────────────────────────────────────
  ✅  ANTHROPIC_API_KEY  (sk-ant-api03…)
  ✅  BUTTERBASE_API_KEY  (bb_sk_cf7326…)

──────────────────────────────────────────────────────
  2 · Butterbase  (append-only evidence trail)
──────────────────────────────────────────────────────
  ✅  claims                 (0 rows)
  ✅  facts                  (0 rows)
  ✅  frames                 (0 rows)
  ✅  fault_analyses         (0 rows)
  ✅  monitoring_events      (0 rows)

──────────────────────────────────────────────────────
  3 · Anthropic  (LLM Integration · latency SLA)
──────────────────────────────────────────────────────
  ✅  model      claude-sonnet-4-6
  ✅  latency    1434 ms  (SLA < 5000 ms)
  ✅  tokens     18
  ✅  cost       $0.000174

──────────────────────────────────────────────────────
  4 · YOLO Weights
──────────────────────────────────────────────────────
  ⚠️   yolo11x.pt not found — auto-downloads on first run

──────────────────────────────────────────────────────
  5 · MLOps Stack  (retry · validation · monitoring)
──────────────────────────────────────────────────────
  ✅  tenacity         retry with exponential backoff
  ✅  pydantic         structured output validation
  ✅  python-dotenv    secrets management

======================================================
  ✅  ALL SYSTEMS GO  —  ready to demo
======================================================
```

**Troubleshooting health check failures:**
- **401 Unauthorized:** Check API key format (`sk-ant-api03-…` / `bb_sk_…`)
- **Connection timeout:** Verify firewall allows HTTPS to `api.anthropic.com`
- **ImportError:** Run `pip install -r requirements.txt` again
- **ModuleNotFoundError:** Ensure you're using Python 3.11+ (`python --version`)

### Step 4: Launch Demo

```bash
python app.py
```

**Output:**
```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

**On first video upload:**
- YOLO11x weights (`yolo11x.pt`, ~130 MB) auto-download from Ultralytics
- Cached in `~/.cache/ultralytics/` — reused for all future runs
- Switch to `yolo11n.pt` (45 MB) in `.env` for faster CPU inference

---

## Production Deployment

### Containerization (Docker)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860')"

EXPOSE 7860

CMD ["python", "app.py"]
```

**Build & run:**
```bash
docker build -t eyewitness:latest .
docker run -p 7860:7860 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e BUTTERBASE_API_KEY=$BUTTERBASE_API_KEY \
  eyewitness:latest
```

### Render.com Deployment (render.yaml included)

```bash
# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/eyewitness.git
git push -u origin main

# Connect to Render.com
# → New Web Service → Connect repository
# → Uses render.yaml automatically
# → Set env vars in dashboard: ANTHROPIC_API_KEY, BUTTERBASE_API_KEY
```

**render.yaml highlights:**
```yaml
services:
  - type: web
    name: eyewitness
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PORT
        value: 7860
      - key: ANTHROPIC_API_KEY
        sync: false  # Set in dashboard
      - key: BUTTERBASE_API_KEY
        sync: false
```

### Kubernetes Deployment (Production-Grade)

**k8s/deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eyewitness
spec:
  replicas: 3
  selector:
    matchLabels:
      app: eyewitness
  template:
    metadata:
      labels:
        app: eyewitness
    spec:
      containers:
      - name: eyewitness
        image: your-registry/eyewitness:v1
        ports:
        - containerPort: 7860
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: anthropic
        - name: BUTTERBASE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: butterbase
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /
            port: 7860
          initialDelaySeconds: 30
          periodSeconds: 10
```

**Create secret:**
```bash
kubectl create secret generic api-keys \
  --from-literal=anthropic=$ANTHROPIC_API_KEY \
  --from-literal=butterbase=$BUTTERBASE_API_KEY
```

**Deploy:**
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## Model Monitoring & Observability

### Built-In Metrics (Logged per Request)

The system tracks **MLOps metrics** automatically:

| Metric | Source | Threshold |
|--------|--------|-----------|
| **Latency (ms)** | Time from upload → report | p95 < 5000 ms |
| **Token count** | Claude API response | Log for cost projection |
| **Cost (USD)** | Input: $0.003/1K, Output: $0.015/1K | Alert if >$1/analysis |
| **Fallback rate** | Pydantic validation failures | <5% |
| **FPS** | Video processing speed | >15 fps target |

**View in report:**
```markdown
### 📊 MLOps Metrics
| Metric        | Value       |
|---------------|-------------|
| Input tokens  | 12,456      |
| Output tokens | 823         |
| Cost          | $0.00493    |
| Fallback      | No ✅       |
| FPS           | 24          |
```

### Drift Detection (Future Enhancement)

**Add monitoring script:**
```python
# scripts/monitoring_report.py (already included)
python scripts/monitoring_report.py --days 7
```

**Tracks:**
- Input distribution shifts (video resolution, duration)
- Prediction confidence trends
- Error rate spikes
- Cost anomalies

**Alert thresholds (to implement):**
```python
ALERT_THRESHOLDS = {
    "p95_latency_ms": 5000,      # Critical: LLM timeout
    "error_rate_pct": 1.0,       # Critical: >1% failures
    "cost_per_req_usd": 0.01,    # Warning: 10x expected cost
    "fallback_rate_pct": 5.0,    # Warning: Pydantic failures
}
```

### Integration with External Monitoring

**Prometheus + Grafana:**
```python
# Add to app.py
from prometheus_client import Counter, Histogram, start_http_server

analysis_requests = Counter('eyewitness_requests_total', 'Total analyses')
analysis_latency = Histogram('eyewitness_latency_seconds', 'Analysis latency')
analysis_cost = Histogram('eyewitness_cost_usd', 'Analysis cost')

# In analyze_clip():
with analysis_latency.time():
    result = analyze_clip(video_path)
analysis_cost.observe(result.hypothesis.cost_usd)
analysis_requests.inc()
```

**Start Prometheus exporter:**
```python
start_http_server(8000)  # Metrics at :8000/metrics
```

---

## API Integration

### Programmatic Usage (Python SDK)

```python
from video_pipeline import analyze_clip
from butterbase_client import ButterbaseClient

# Run analysis
result = analyze_clip("dashcam_footage.mp4")

# Access structured outputs
print(f"Run ID: {result.run_id}")
for fact in result.vehicle_facts:
    print(f"Vehicle #{fact.vehicle_id}: {fact.speed_kph_est} kph")

for avoid in result.avoidability:
    print(f"Vehicle #{avoid.vehicle_id}: {avoid.verdict}")

print(f"Fault hypothesis: {result.hypothesis.fault_reason}")
print(f"Cost: ${result.hypothesis.cost_usd:.5f}")

# Store in Butterbase
bb = ButterbaseClient()
bb.append_facts(result.run_id, result.vehicle_facts)
bb.append_hypothesis(result.run_id, result.hypothesis)
```

### REST API (Future Enhancement)

**Proposed endpoint design:**
```bash
POST /api/v1/analyze
Content-Type: multipart/form-data

{
  "video": <file>,
  "metadata": {
    "claim_id": "CLM-2026-001",
    "timestamp": "2026-06-20T10:00:00Z"
  }
}

# Response
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "vehicle_facts": [...],
  "avoidability": [...],
  "hypothesis": {
    "fault_vehicle_id": 1,
    "confidence": 0.85,
    "cost_usd": 0.00493
  },
  "keyframes": [
    {"stage": "overview", "url": "https://..."},
    {"stage": "pre_impact", "url": "https://..."}
  ]
}
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| **ImportError: No module named 'cv2'** | opencv-python not installed | `pip install opencv-python` |
| **anthropic.APIConnectionError** | API key invalid or expired | Check `.env` file, regenerate key |
| **Pydantic ValidationError** | Claude output doesn't match schema | Check logs for fallback usage |
| **YOLO model not found** | Auto-download failed (firewall?) | Manually download from [ultralytics.com](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) → place in project root |
| **Gradio crashes on video upload** | Video codec not supported | Convert to H.264: `ffmpeg -i input.mp4 -c:v libx264 output.mp4` |
| **High latency (>10s)** | Claude API overloaded | Retry automatically via tenacity; check status.anthropic.com |

### Debug Mode

**Enable verbose logging:**
```bash
export PYTHONUNBUFFERED=1
python app.py 2>&1 | tee demo.log
```

**Check Butterbase writes:**
```python
# Test database connection
from butterbase_client import ButterbaseClient
bb = ButterbaseClient()
print(bb.query_all_claims())  # Should return empty list or existing claims
```

### Performance Optimization

**Speed up inference (CPU):**
```bash
# Switch to smaller YOLO model
echo "YOLO_MODEL=yolo11n.pt" >> .env
```

**GPU acceleration (if available):**
```python
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# YOLO auto-detects GPU
# Check with: python -c "import torch; print(torch.cuda.is_available())"
```

---

## Next Steps

### Production Checklist

- [ ] **API keys secured:** Stored in secrets manager (AWS Secrets Manager, Vault)
- [ ] **Rate limiting configured:** 50 req/min Anthropic limit monitored
- [ ] **Cost alerts set up:** PagerDuty when >$100/day
- [ ] **Drift monitoring enabled:** `scripts/monitoring_report.py` cron job
- [ ] **Backup strategy:** Butterbase evidence trail (daily exports)
- [ ] **Incident response:** Playbook for LLM API outages documented
- [ ] **Load testing:** Validated 100 concurrent analyses
- [ ] **Model versioning:** `model_version` tracked per analysis

### Testing Workflow

1. **Smoke test:** Upload sample video → verify 3-layer analysis
2. **Butterbase verification:** Check all 5 tables populated correctly
3. **Cost calculation:** Measure $/analysis for your video types
4. **Latency SLA:** Confirm p95 < 5s under load
5. **Fallback testing:** Intentionally corrupt Claude output → verify graceful degradation

### Model Evolution

**A/B testing Claude versions:**
```python
# In config.py, add:
CLAUDE_MODELS = {
    "control": "claude-sonnet-4-6",
    "treatment": "claude-3-5-sonnet-latest"
}

# Randomly assign 50/50, log model_version in Butterbase
# Compare: cost, latency, confidence, override rate
```

**Automated retraining triggers:**
- **Performance drop:** Override rate >10% for a specific fault type
- **Data drift:** New video characteristics (night footage, rain)
- **Cost anomaly:** Token usage >2x baseline

---

## Support & Resources

- **GitHub Issues:** Report bugs, request features
- **Anthropic Status:** [status.anthropic.com](https://status.anthropic.com)
- **Butterbase Docs:** [butterbase.ai/docs](https://butterbase.ai/docs)
- **MLOps References:** See skill document `SKILL (4).md`

---

**Built with ❤️ for production ML systems**  
*Version: eyewitness-v1*
