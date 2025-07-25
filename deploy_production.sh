#!/bin/bash

# Digital Twin Generator - Production Deployment Script
# Optimized for RunPod A10G+ GPU instances

set -e

echo "🚀 Digital Twin Generator - Production Deployment"
echo "================================================"
echo "Target: RunPod A10G+ GPU Instance"
echo "Versions: diffusers==0.25.0, huggingface_hub==0.25.0"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

# Check if running on RunPod
if [[ "$HOSTNAME" == *"runpod"* ]] || [[ "$HOSTNAME" == *"pod"* ]]; then
    echo -e "${BLUE}🎯 RunPod environment detected${NC}"
    RUNPOD=true
else
    echo -e "${YELLOW}⚠️  Not running on RunPod - some optimizations may not apply${NC}"
    RUNPOD=false
fi

# Check GPU availability
echo -e "${BLUE}🔍 Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    echo "GPU: $GPU_INFO"
    print_status 0 "GPU detected"
else
    echo -e "${RED}❌ No NVIDIA GPU detected${NC}"
    exit 1
fi

# Check system resources
echo -e "${BLUE}💾 Checking system resources...${NC}"
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_DISK=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
echo "Total RAM: ${TOTAL_MEM}GB"
echo "Available Disk: ${AVAILABLE_DISK}GB"

if [ "$TOTAL_MEM" -lt 16 ]; then
    echo -e "${YELLOW}⚠️  Warning: Less than 16GB RAM detected${NC}"
fi

if [ "$AVAILABLE_DISK" -lt 50 ]; then
    echo -e "${YELLOW}⚠️  Warning: Less than 50GB disk space available${NC}"
fi

# Update system packages
echo -e "${BLUE}📦 Updating system packages...${NC}"
apt-get update -qq
apt-get install -y -qq curl wget git python3-pip python3-venv

# Install system dependencies
echo -e "${BLUE}🔧 Installing system dependencies...${NC}"
apt-get install -y -qq \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6

print_status $? "System dependencies installed"

# Create virtual environment
echo -e "${BLUE}🐍 Setting up Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install exact production versions
echo -e "${BLUE}📚 Installing Python dependencies (Production Versions)...${NC}"

# Core AI/ML Dependencies (Exact versions as specified)
pip install torch>=2.0.0 torchvision>=0.15.0
pip install diffusers==0.25.0
pip install huggingface_hub==0.25.0
pip install transformers>=4.36.2
pip install accelerate>=0.27.2
pip install peft==0.7.1

# Computer Vision & Image Processing
pip install opencv-python>=4.8.0
pip install Pillow>=9.5.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install mediapipe>=0.10.0

# Face Analysis & Enhancement
pip install facexlib>=0.3.0
pip install basicsr>=1.4.2
pip install gfpgan>=1.3.8
pip install insightface>=0.7.3
pip install onnxruntime>=1.15.0

# Web Framework
pip install flask>=2.3.0
pip install flask-cors>=4.0.0
pip install werkzeug>=2.3.0

# Utilities
pip install requests>=2.31.0
pip install tqdm>=4.65.0
pip install safetensors>=0.3.0
pip install xformers>=0.0.20
pip install datasets>=2.10.0
pip install psutil>=5.9.0

# Optional: CodeFormer (compatible version)
pip install codeformer==0.0.11

print_status $? "Python dependencies installed"

# Download models
echo -e "${BLUE}📥 Downloading AI models...${NC}"
chmod +x download_models.sh
./download_models.sh

print_status $? "Models downloaded"

# Create necessary directories
echo -e "${BLUE}📁 Creating application directories...${NC}"
mkdir -p selfies output models web/static web/templates

# Set up environment variables
echo -e "${BLUE}⚙️  Setting up environment variables...${NC}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=/workspace/.cache/huggingface

# Test model loading
echo -e "${BLUE}🧪 Testing model loading...${NC}"
python -c "
import torch
import diffusers
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

print_status $? "Model loading test completed"

# Create startup script
echo -e "${BLUE}🚀 Creating startup script...${NC}"
cat > start_app.sh << 'EOF'
#!/bin/bash

# Digital Twin Generator Startup Script
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=/workspace/.cache/huggingface

# Activate virtual environment
source venv/bin/activate

# Start the Flask application
echo "🎯 Starting Digital Twin Generator..."
echo "🌐 Web interface: http://0.0.0.0:5000"
echo "📊 Health check: http://0.0.0.0:5000/health"
echo "🧪 Test page: http://0.0.0.0:5000/test"

python app.py
EOF

chmod +x start_app.sh

# Create systemd service (if on RunPod)
if [ "$RUNPOD" = true ]; then
    echo -e "${BLUE}🔧 Creating systemd service...${NC}"
    cat > /etc/systemd/system/digital-twin-generator.service << EOF
[Unit]
Description=Digital Twin Generator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
Environment=HF_HOME=/workspace/.cache/huggingface
ExecStart=/workspace/start_app.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable digital-twin-generator.service
    print_status $? "Systemd service created"
fi

# Create health check script
echo -e "${BLUE}🏥 Creating health check script...${NC}"
cat > health_check.sh << 'EOF'
#!/bin/bash

# Health check for Digital Twin Generator
curl -f http://localhost:5000/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Service is healthy"
    exit 0
else
    echo "❌ Service is unhealthy"
    exit 1
fi
EOF

chmod +x health_check.sh

# Create monitoring script
echo -e "${BLUE}📊 Creating monitoring script...${NC}"
cat > monitor.sh << 'EOF'
#!/bin/bash

# Monitoring script for Digital Twin Generator
echo "=== Digital Twin Generator Status ==="
echo "Time: $(date)"
echo ""

# Service status
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Service: Running"
else
    echo "❌ Service: Not responding"
fi

# GPU status
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r util mem_used mem_total; do
        echo "GPU Utilization: ${util}%"
        echo "GPU Memory: ${mem_used}MB / ${mem_total}MB"
    done
fi

# System resources
echo ""
echo "=== System Resources ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df / | awk 'NR==2 {print $5}')"

# Active jobs
echo ""
echo "=== Active Jobs ==="
curl -s http://localhost:5000/jobs | jq -r '.jobs[] | "\(.id): \(.status) (\(.progress)%)"' 2>/dev/null || echo "No active jobs"
EOF

chmod +x monitor.sh

# Create curl test examples
echo -e "${BLUE}📋 Creating curl test examples...${NC}"
cat > curl_examples.md << 'EOF'
# Digital Twin Generator - cURL Examples

## Health Check
```bash
curl http://localhost:5000/health
```

## Upload 15+ Selfies
```bash
curl -F "files=@selfie1.jpg" \
     -F "files=@selfie2.jpg" \
     -F "files=@selfie3.jpg" \
     ... \
     -F "files=@selfie15.jpg" \
     http://localhost:5000/upload
```

## Generate Avatar
```bash
curl -H "Content-Type: application/json" \
     -d '{
         "prompt": "a realistic cinematic portrait of a woman in cyberpunk city background",
         "num_images": 1,
         "quality_mode": "high_fidelity"
     }' \
     http://localhost:5000/generate
```

## Check Job Status
```bash
curl http://localhost:5000/status/JOB_ID_HERE
```

## Download Generated Image
```bash
curl http://localhost:5000/download/JOB_ID_HERE/avatar_001.png
```

## System Status
```bash
curl http://localhost:5000/system-status
```
EOF

print_status $? "Documentation created"

# Final setup
echo -e "${BLUE}🎯 Final setup...${NC}"

# Make scripts executable
chmod +x *.sh

# Create a simple test
echo -e "${BLUE}🧪 Running quick test...${NC}"
python -c "
from app import app
print('✅ Flask app imports successfully')
"

print_status $? "Quick test completed"

echo ""
echo -e "${GREEN}🎉 Production Deployment Completed!${NC}"
echo ""
echo "📋 Next Steps:"
echo "1. Start the application: ./start_app.sh"
echo "2. Test the API: ./test_api.sh"
echo "3. Monitor the service: ./monitor.sh"
echo "4. Access web interface: http://0.0.0.0:5000"
echo "5. Test page: http://0.0.0.0:5000/test"
echo ""
echo "🔧 Configuration:"
echo "• Models: /models/"
echo "• Selfies: /selfies/"
echo "• Output: /output/"
echo "• Logs: Check console output"
echo ""
echo "🚀 For production use:"
echo "• Use systemd: systemctl start digital-twin-generator"
echo "• Monitor: systemctl status digital-twin-generator"
echo "• Logs: journalctl -u digital-twin-generator -f"
echo ""
echo -e "${BLUE}🎯 Ready for client demo!${NC}" 