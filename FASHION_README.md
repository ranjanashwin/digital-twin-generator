# Fashion Digital Twin Generator

A comprehensive AI-powered fashion content creation platform that generates photorealistic digital twins and fashion photos using advanced AI models including IPAdapter, Stable Diffusion XL, ControlNet, and VITON-HD.

## 🚀 Features

### Core Capabilities
- **Digital Twin Generation**: Create photorealistic avatars from 15+ selfies using IPAdapter + SDXL
- **Fashion Photo Generation**: Generate fashion content with clothing try-on using VITON-HD + ControlNet
- **3-Step Workflow**: Streamlined process from selfie upload to final fashion photo
- **High-Quality Output**: Multiple quality modes (Standard, High Fidelity, Ultra Fidelity)
- **Real-time Progress**: Live progress tracking and status updates

### AI Models Integration
- **IPAdapter FaceID**: Identity preservation and consistency
- **Stable Diffusion XL**: High-resolution image generation
- **ControlNet**: Pose and depth conditioning
- **VITON-HD**: Virtual clothing try-on
- **GFPGAN/CodeFormer**: Face enhancement and restoration
- **MediaPipe**: Pose estimation and facial landmarks

### Fashion Workflow Steps

#### Step 1: Upload Selfies & Generate Digital Twin
- Upload ZIP file containing 15+ selfies
- Choose avatar style (Fashion Portrait, Street Style, Studio Fashion, Editorial)
- Add custom prompt for personalized results
- Select quality mode for generation speed/quality trade-off
- Generate photorealistic digital twin

#### Step 2: Upload Clothing & Scene Description
- Upload clothing image (shirt, dress, etc.)
- Describe desired scene and lighting (e.g., "golden hour at Paris rooftop")
- Select quality mode for fashion photo generation
- Process clothing with VITON-HD for virtual try-on

#### Step 3: View & Download Results
- Preview generated digital twin and fashion photo
- Download high-resolution images
- Option to regenerate or start over

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (A10G+ recommended)
- 16GB+ VRAM for optimal performance
- 50GB+ free disk space for models

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fashion-digital-twin-generator.git
cd fashion-digital-twin-generator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download AI models**
```bash
python download_models.py
```

4. **Start the application**
```bash
python app.py
```

5. **Access the web interface**
```
http://localhost:5000
```

### Production Deployment (RunPod)

1. **Create RunPod instance**
   - GPU: A10G or better
   - RAM: 32GB+
   - Storage: 100GB+

2. **Deploy using the provided script**
```bash
chmod +x deploy_production.sh
./deploy_production.sh
```

3. **Access via RunPod URL**
   - HTTP Port: 5000
   - HTTPS Port: 443

## 📁 Project Structure

```
fashion-digital-twin-generator/
├── app.py                          # Main Flask application
├── generate_twin.py                # Core avatar generation logic
├── requirements.txt                # Python dependencies
├── config.py                      # Configuration settings
├── run.py                         # Application entry point
├── download_models.py             # Model downloader
├── deploy_production.sh           # Production deployment script
├── PRODUCTION_README.md           # Production deployment guide
├── FASHION_README.md              # This file
│
├── utils/                         # Utility modules
│   ├── ipadapter_manager.py      # IPAdapter integration
│   ├── viton_integration.py      # VITON-HD integration
│   ├── controlnet_integration.py # ControlNet integration
│   ├── image_validator.py        # Image validation
│   ├── resource_manager.py       # Resource management
│   └── model_loader.py           # Model loading utilities
│
├── web/                          # Frontend assets
│   ├── templates/
│   │   ├── fashion_workflow.html # Main workflow interface
│   │   ├── index.html            # Legacy interface
│   │   └── demo.html             # Demo page
│   ├── static/
│   │   ├── css/
│   │   │   ├── fashion_styles.css # Fashion workflow styles
│   │   │   └── styles.css        # Legacy styles
│   │   └── js/
│   │       ├── fashion_workflow.js # Fashion workflow logic
│   │       └── script.js         # Legacy JavaScript
│   └── static/images/            # Static images
│
├── models/                       # AI model storage
│   ├── sdxl/                    # Stable Diffusion XL models
│   ├── ipadapter/               # IPAdapter models
│   ├── controlnet/              # ControlNet models
│   └── viton_hd/               # VITON-HD models
│
├── output/                      # Generated outputs
├── selfies/                     # Uploaded selfies (temporary)
└── tests/                       # Test files
```

## 🎨 Fashion Workflow Interface

### Step 1: Upload Selfies
- **Drag & Drop**: Upload ZIP file with 15+ selfies
- **Avatar Styles**: Choose from Fashion Portrait, Street Style, Studio Fashion, Editorial
- **Custom Prompts**: Add personalized style descriptions
- **Quality Modes**: Standard (3-5 min), High Fidelity (5-8 min), Ultra Fidelity (8-12 min)

### Step 2: Upload Clothing & Scene
- **Clothing Upload**: Drag & drop clothing image
- **Scene Description**: Describe lighting, location, mood
- **Quality Selection**: Choose generation quality for fashion photo

### Step 3: Results & Download
- **Preview Results**: View generated digital twin and fashion photo
- **Download Options**: High-resolution image downloads
- **Regeneration**: Generate new variations
- **Restart**: Begin new workflow

## 🔧 Configuration

### Quality Modes
```python
FASHION_QUALITY_MODES = {
    "standard": {
        "resolution": (768, 768),
        "steps": 20,
        "guidance_scale": 7.0,
        "estimated_time": "3-5 minutes"
    },
    "high_fidelity": {
        "resolution": (1024, 1024),
        "steps": 30,
        "guidance_scale": 8.0,
        "estimated_time": "5-8 minutes"
    },
    "ultra_fidelity": {
        "resolution": (1024, 1024),
        "steps": 50,
        "guidance_scale": 8.5,
        "estimated_time": "8-12 minutes"
    }
}
```

### Avatar Styles
```python
FASHION_AVATAR_STYLES = {
    "fashion_portrait": {
        "name": "Fashion Portrait",
        "description": "High-fashion editorial style",
        "prompt_template": "a high-fashion portrait of the person, editorial lighting, professional photography, fashion magazine style, sharp details"
    },
    "street_style": {
        "name": "Street Style",
        "description": "Casual street fashion look",
        "prompt_template": "a street style photo of the person, natural lighting, urban background, candid fashion photography"
    },
    # ... more styles
}
```

## 🚀 API Endpoints

### Core Endpoints
- `POST /upload-selfies` - Upload selfies and generate avatar
- `POST /upload-clothing-scene` - Upload clothing and generate fashion photo
- `GET /status/<job_id>` - Get generation status
- `GET /download/<job_id>/<filename>` - Download generated images
- `GET /session/<session_id>` - Get session results

### Configuration Endpoints
- `GET /fashion-styles` - Get available avatar styles
- `GET /fashion-quality-modes` - Get quality mode configurations
- `GET /health` - Health check
- `GET /system-status` - System resource status

## 🎯 Use Cases

### Fashion Content Creation
- **E-commerce**: Generate model photos with different clothing
- **Social Media**: Create consistent avatar for brand presence
- **Marketing**: Produce fashion campaign imagery
- **Personal Branding**: Professional headshots and lifestyle photos

### Technical Applications
- **Virtual Try-On**: Test clothing on digital twins
- **Style Transfer**: Apply different fashion styles
- **Pose Generation**: Create diverse pose variations
- **Lighting Simulation**: Test different lighting scenarios

## 🔍 Model Integration Details

### IPAdapter FaceID
- **Purpose**: Identity preservation across generations
- **Integration**: Batch averaging of multiple selfie embeddings
- **Benefits**: Consistent facial features and expressions

### VITON-HD
- **Purpose**: Virtual clothing try-on
- **Components**: Segmentation, Geometric Matching, Try-On Module
- **Output**: Realistic clothing fitting and draping

### ControlNet
- **Purpose**: Pose and depth conditioning
- **Types**: Pose estimation, depth mapping, edge detection
- **Benefits**: Precise pose control and scene composition

### Stable Diffusion XL
- **Purpose**: High-resolution image generation
- **Resolution**: Up to 1024x1024
- **Quality**: Photorealistic output with fine details

## 🛡️ Security & Privacy

### Data Handling
- **Temporary Storage**: Uploaded files stored temporarily
- **Session Isolation**: Each session has isolated file storage
- **Auto Cleanup**: Automatic cleanup of old sessions
- **No Persistence**: No permanent storage of user images

### Model Safety
- **Content Filtering**: Built-in content safety filters
- **Quality Validation**: Image quality and content validation
- **Error Handling**: Graceful error handling and recovery

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Test Specific Components
```bash
# Test image validation
python test_validation_system.py

# Test model integration
python test_lora_integration.py

# Test quality modes
python test_quality_modes.py
```

## 📊 Performance

### Hardware Requirements
- **Minimum**: RTX 3080 (10GB VRAM)
- **Recommended**: A10G (24GB VRAM)
- **Optimal**: A100 (40GB+ VRAM)

### Generation Times
- **Standard Mode**: 3-5 minutes
- **High Fidelity**: 5-8 minutes
- **Ultra Fidelity**: 8-12 minutes

### Memory Usage
- **Model Loading**: ~8GB VRAM
- **Generation**: ~12-16GB VRAM
- **Peak Usage**: ~20GB VRAM

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Run tests
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Write unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IPAdapter**: For identity preservation technology
- **Stable Diffusion XL**: For high-quality image generation
- **ControlNet**: For pose and depth conditioning
- **VITON-HD**: For virtual clothing try-on
- **Hugging Face**: For model hosting and distribution

## 📞 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions
- **Documentation**: Check the wiki for detailed guides

## 🔄 Updates

### Latest Features
- ✅ Fashion workflow with 3-step process
- ✅ VITON-HD integration for clothing try-on
- ✅ ControlNet integration for pose conditioning
- ✅ Real-time progress tracking
- ✅ Multiple quality modes
- ✅ Session management
- ✅ Auto-cleanup system

### Roadmap
- 🔄 Batch processing
- 🔄 Advanced pose control
- 🔄 Multiple clothing items
- 🔄 Video generation
- 🔄 Mobile app
- 🔄 API rate limiting
- 🔄 User authentication 