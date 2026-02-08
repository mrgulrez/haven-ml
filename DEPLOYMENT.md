# Deployment Guide

Complete guide for deploying the Empathy System to production.

---

## Deployment Options

### Option 1: Local Development (Recommended for Testing)
**Best for**: Initial testing, development, demos  
**Requirements**: Your local machine with GPU  
**Setup Time**: 15-30 minutes  
**See**: [SETUP.md](SETUP.md)

---

### Option 2: Docker (Local or Cloud)
**Best for**: Consistent environments, easy deployment  
**Requirements**: Docker installed  
**Setup Time**: 10 minutes  
**Cost**: Free (local) or $0.50-2/hour (cloud)

---

### Option 3: Cloud GPU Providers
**Best for**: Production deployment, scalability  
**Providers**: RunPod, Lambda Labs, Vast.ai  
**Setup Time**: 20-30 minutes  
**Cost**: $0.30-1.50/hour depending on GPU

---

## Quick Start: Docker Deployment

### Prerequisites
```bash
# Install Docker
# Windows: Download from docker.com
# Linux: sudo apt-get install docker.io docker-compose
```

### Step 1: Build Docker Image

**CPU Version** (for testing):
```bash
cd d:/Pet/ML/empathy-system

# Build image
docker build -t empathy-system:cpu -f Dockerfile .

# Run container
docker run -it --rm \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  empathy-system:cpu
```

**GPU Version** (for production):
```bash
# Ensure NVIDIA Docker runtime is installed
# Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Build GPU image
docker build -t empathy-system:gpu -f Dockerfile.gpu .

# Run with GPU
docker run -it --rm \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  empathy-system:gpu
```

---

### Step 2: Using Docker Compose (Easier)

**CPU Version**:
```bash
docker-compose up -d
```

**GPU Version**:
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

**Check logs**:
```bash
docker-compose logs -f empathy-agent
```

**Stop**:
```bash
docker-compose down
```

---

## Cloud Deployment: RunPod

RunPod offers affordable GPU instances with Docker support.

### Step 1: Create RunPod Account
1. Go to [runpod.io](https://runpod.io)
2. Sign up and add billing

### Step 2: Deploy Pod

**Option A: Using Docker Image**:
```bash
# Tag and push to Docker Hub (one-time)
docker tag empathy-system:gpu yourusername/empathy-system:latest
docker push yourusername/empathy-system:latest
```

Then in RunPod:
1. Click "Deploy"
2. Select GPU (RTX 3090 or A4000 recommended)
3. Choose "Docker Image"
4. Enter: `yourusername/empathy-system:latest`
5. Set ports: 8080
6. Deploy!

**Option B: Using RunPod Template**:
1. Select "PyTorch" template
2. GPU: RTX 3090 ($0.34/hr) or A4000 ($0.44/hr)
3. Volume: 50GB
4. Deploy

Then SSH into pod:
```bash
# Clone repo
git clone <your-repo-url>
cd empathy-system

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

### Step 3: Access Your Instance

RunPod provides:
- **Public IP**: For API access
- **SSH Access**: For debugging
- **Jupyter**: For development

Access at: `http://<runpod-ip>:8080`

**Cost Estimate**: $0.34-0.70/hour depending on GPU

---

## Cloud Deployment: Lambda Labs

Lambda Labs offers high-performance GPUs optimized for ML.

### Step 1: Create Account
1. Go to [lambdalabs.com](https://lambdalabs.com/service/gpu-cloud)
2. Sign up

### Step 2: Launch Instance

1. Click "Launch Instance"
2. Select GPU: A4000 or A6000
3. Select region
4. Upload SSH key
5. Launch

### Step 3: Deploy Application

SSH into instance:
```bash
ssh ubuntu@<lambda-ip>

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker

# Clone and deploy
git clone <your-repo-url>
cd empathy-system
sudo docker-compose -f docker-compose.gpu.yml up -d
```

**Cost Estimate**: $0.50-1.10/hour depending on GPU

---

## Cloud Deployment: Vast.ai

Vast.ai offers the cheapest GPU rentals (community GPUs).

### Step 1: Search for Instance

1. Go to [vast.ai](https://vast.ai)
2. Create account
3. Click "Search" → filter by:
   - GPU: RTX 3090 or better
   - CUDA: 11.8 or 12.1
   - Disk: 50GB+
   - Sort by: Price (low to high)

### Step 2: Rent Instance

1. Click "Rent"
2. Choose "Docker Image": `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`
3. Set on-start script:
```bash
git clone <your-repo-url> /workspace/empathy-system
cd /workspace/empathy-system
pip install -r requirements.txt
python main.py
```
4. Start instance

**Cost Estimate**: $0.20-0.50/hour (community GPUs)

---

## Production Checklist

Before deploying to production:

### Security
- [ ] Change all default passwords
- [ ] Enable HTTPS/SSL
- [ ] Set up firewall rules
- [ ] Configure API authentication
- [ ] Enable encryption for user data

### Performance
- [ ] Test with real models (not mocks)
- [ ] Profile latency on target GPU
- [ ] Set up monitoring (CPU, GPU, RAM usage)
- [ ] Configure logging
- [ ] Test with multiple concurrent users

### Reliability
- [ ] Set up automatic restarts
- [ ] Configure health checks
- [ ] Set up backup system for user profiles
- [ ] Test failover scenarios
- [ ] Set up alerting

### Privacy
- [ ] Review data collection policies
- [ ] Implement "forget me" feature
- [ ] Set up automatic data deletion (24h)
- [ ] Add camera/mic mute controls
- [ ] Document privacy practices

---

## Monitoring Setup

### Simple Monitoring (Docker)

Add to `docker-compose.yml`:
```yaml
  monitoring:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### Application Metrics

Add to your code:
```python
from prometheus_client import Counter, Histogram, start_http_server

# Start metrics server
start_http_server(8000)

# Track requests
requests = Counter('empathy_requests_total', 'Total requests')
latency = Histogram('empathy_latency_seconds', 'Request latency')
```

---

## Scaling Considerations

### Horizontal Scaling

For multiple users:
1. Deploy multiple instances
2. Use load balancer (nginx, HAProxy)
3. Share Redis for session data
4. Separate model servers from logic

### Kubernetes (Advanced)

For large-scale deployment:
```bash
# Build and push image
docker build -t gcr.io/your-project/empathy-system:v1 -f Dockerfile.gpu .
docker push gcr.io/your-project/empathy-system:v1

# Deploy to GKE/EKS
kubectl apply -f k8s/deployment.yaml
```

---

## Cost Optimization

### Tips to Reduce Costs

1. **Use Spot/Preemptible Instances**: 50-70% cheaper
2. **Auto-shutdown**: Stop instances when idle
3. **Smaller Models**: Use quantized models (4-bit)
4. **Batch Processing**: Process multiple requests together
5. **CPU for Some Tasks**: Use CPU for VAD, STT; GPU for LLM only

### Estimated Monthly Costs

| Setup | GPU | Cost/Hour | 24/7 Cost/Month | 8h/day Cost/Month |
|-------|-----|-----------|-----------------|-------------------|
| RunPod RTX 3090 | 24GB | $0.34 | $245 | $82 |
| Lambda A4000 | 16GB | $0.60 | $432 | $144 |
| Vast.ai RTX 3090 | 24GB | $0.25 | $180 | $60 |
| Local (Your GPU) | - | $0 | $0 | $0 |

**Recommendation**: Start local, then Vast.ai for production testing, then RunPod/Lambda for production.

---

## Troubleshooting Deployment

### Docker Build Fails
```bash
# Check Docker is running
docker info

# Check disk space
df -h

# Rebuild without cache
docker build --no-cache -t empathy-system:gpu -f Dockerfile.gpu .
```

### GPU Not Detected in Container
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory
- Use smaller batch sizes
- Use quantized models (4-bit vs 16-bit)
- Reduce context length
- Use gradient checkpointing

### High Latency
- Check GPU utilization: `nvidia-smi`
- Profile code: Use `@timeit` decorators
- Enable model caching
- Use smaller models

---

## Next Steps

1. ✅ Test locally (see [SETUP.md](SETUP.md))
2. ✅ Build Docker image
3. ✅ Deploy to cloud (RunPod recommended)
4. ⚠️ Add monitoring
5. ⚠️ Production hardening
6. ⚠️ Scale as needed
