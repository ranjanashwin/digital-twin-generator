# Digital Twin Generator - Setup Guide

## Quick Start

### Option 1: Simple Run (Recommended)
```bash
python run.py
```
This will automatically:
- Install dependencies
- Download models
- Run tests
- Start the web application

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models
python download_models.py

# 3. Test installation
python test_installation.py

# 4. Start web app
python app.py
```

## Deployment Options

### Local Development
```bash
# Clone and setup
git clone <repository-url>
cd digital-twin-generator
python run.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t digital-twin-generator .
docker run --gpus all -p 5000:5000 digital-twin-generator
```

### RunPod Deployment
1. Upload the project to RunPod
2. Install dependencies: `pip install -r requirements.txt`
3. Download models: `python download_models.py`
4. Start the application: `python app.py`
5. Access via the provided URL

## GPU Requirements

### Minimum Requirements
- **GPU**: RTX 3080 (10GB VRAM) or better
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space for models

### Recommended Requirements
- **GPU**: A10G (24GB VRAM) or better
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space

## Model Downloads

The following models will be downloaded automatically:
- **Stable Diffusion XL Base**: ~6GB
- **Stable Diffusion XL Refiner**: ~6GB
- **IPAdapter FaceID**: ~2GB
- **GFPGAN**: ~300MB
- **CodeFormer**: ~200MB
- **InsightFace**: ~100MB

Total download size: ~15GB

## Usage

### Web Interface
1. Open http://localhost:5000
2. Upload a ZIP file containing 15+ selfies
3. Select generation style
4. Click "Generate Digital Twin"
5. Download the result

### Command Line
```bash
python generate_twin.py \
  --input_folder /path/to/selfies \
  --output_folder /path/to/output \
  --prompt_style portrait \
  --num_images 1
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config.py`
- Use gradient checkpointing
- Close other GPU applications
- Try with smaller resolution

### Model Download Issues
- Check internet connection
- Use VPN if needed
- Manual download from HuggingFace
- Clear cache: `rm -rf ~/.cache/huggingface`

### Face Detection Issues
- Ensure selfies show clear, front-facing faces
- Minimum 15 selfies recommended
- Good lighting in source images
- Check face detection confidence in config

### Web Interface Issues
- Check if port 5000 is available
- Try different port in `config.py`
- Check firewall settings
- Verify Flask installation

## Configuration

Edit `config.py` to customize:
- **Generation settings**: Resolution, steps, guidance scale
- **IPAdapter settings**: Weight, noise, timing
- **Face enhancement**: Method, quality settings
- **Web interface**: Port, file size limits

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload selfies ZIP
- `GET /status/<job_id>` - Check generation status
- `GET /download/<job_id>/<filename>` - Download result
- `GET /health` - Health check

## File Structure

```
digital-twin-generator/
├── app.py                 # Flask web application
├── generate_twin.py       # Core generation logic
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── run.py               # Easy startup script
├── download_models.py   # Model downloader
├── test_installation.py # Installation tests
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── README.md           # Main documentation
├── SETUP.md            # This setup guide
├── models/             # Downloaded models
│   ├── ip_adapter/    # IPAdapter models
│   ├── sdxl/         # Stable Diffusion XL
│   └── face_enhance/ # Face enhancement models
├── utils/             # Utility functions
│   ├── face_utils.py  # Face processing
│   ├── model_loader.py # Model loading
│   └── ip_adapter_wrapper.py # IPAdapter wrapper
├── web/               # Web interface
│   └── templates/     # HTML templates
├── selfies/          # Uploaded selfies
└── output/           # Generated images
```

## Performance Tips

1. **GPU Memory**: Use A10G+ for best performance
2. **Batch Processing**: Process multiple requests sequentially
3. **Model Caching**: Models are cached in memory after first load
4. **Face Enhancement**: GFPGAN is faster than CodeFormer
5. **Resolution**: 1024x1024 is optimal for quality/speed balance

## Security Notes

- Web interface runs on all interfaces (0.0.0.0)
- No authentication implemented
- File uploads limited to 100MB
- Temporary files auto-cleanup after 1 hour
- Consider reverse proxy for production use

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run `python test_installation.py`
3. Check logs in console output
4. Verify GPU and CUDA installation
5. Ensure sufficient disk space for models 