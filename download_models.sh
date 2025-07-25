#!/bin/bash

# Digital Twin Generator - Model Download Script
# Optimized for cloud GPU deployment
# Supports HuggingFace CLI authentication

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check git
    if ! command_exists git; then
        print_error "Git is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command_exists pip3; then
        print_error "pip3 is not installed"
        exit 1
    fi
    
    print_success "System requirements met"
}

# Function to install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install required packages
    pip3 install --upgrade pip
    pip3 install requests tqdm huggingface-hub
    
    print_success "Python dependencies installed"
}

# Function to setup HuggingFace CLI
setup_huggingface() {
    print_status "Setting up HuggingFace CLI..."
    
    # Install huggingface-cli if not present
    if ! command_exists huggingface-cli; then
        print_status "Installing huggingface-cli..."
        pip3 install huggingface-hub
    fi
    
    # Check if already logged in
    if huggingface-cli whoami >/dev/null 2>&1; then
        print_success "Already logged in to HuggingFace"
        return 0
    fi
    
    # Prompt for HuggingFace token
    print_warning "HuggingFace authentication required for some models"
    print_status "You can get your token from: https://huggingface.co/settings/tokens"
    echo
    read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN
    
    if [ -n "$HF_TOKEN" ]; then
        print_status "Logging in to HuggingFace..."
        echo "$HF_TOKEN" | huggingface-cli login
        print_success "Logged in to HuggingFace"
    else
        print_warning "Skipping HuggingFace login - some models may not download"
    fi
}

# Function to check available storage
check_storage() {
    print_status "Checking available storage..."
    
    # Get available space in GB
    if command_exists df; then
        AVAILABLE_GB=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
        print_status "Available storage: ${AVAILABLE_GB}GB"
        
        if [ "$AVAILABLE_GB" -lt 50 ]; then
            print_warning "Less than 50GB available - ensure sufficient space for models"
        fi
    fi
}

# Function to detect cloud GPU environment
detect_environment() {
    print_status "Detecting environment..."
    
    if [ -n "$CUDA_VISIBLE_DEVICES" ] || [ -n "$GPU" ]; then
        print_success "Cloud GPU environment detected"
        ENVIRONMENT="cloud"
    else
        print_status "Local environment detected"
        ENVIRONMENT="local"
    fi
    
    # Check for common cloud GPU providers
    if [ -f "/proc/cpuinfo" ]; then
        if grep -q "AWS" /proc/cpuinfo 2>/dev/null; then
            print_status "AWS environment detected"
        elif grep -q "Google" /proc/cpuinfo 2>/dev/null; then
            print_status "Google Cloud environment detected"
        elif grep -q "Azure" /proc/cpuinfo 2>/dev/null; then
            print_status "Azure environment detected"
        fi
    fi
}

# Function to optimize for cloud deployment
optimize_for_cloud() {
    if [ "$ENVIRONMENT" = "cloud" ]; then
        print_status "Optimizing for cloud deployment..."
        
        # Set environment variables for better performance
        export HF_HUB_DISABLE_TELEMETRY=1
        export HF_HUB_DISABLE_SYMLINKS_WARNING=1
        
        # Optimize git for large downloads
        git config --global http.postBuffer 524288000
        git config --global core.compression 9
        
        print_success "Cloud optimizations applied"
    fi
}

# Function to run the Python downloader
run_downloader() {
    print_status "Starting model download process..."
    
    # Check if Python script exists
    if [ ! -f "download_models.py" ]; then
        print_error "download_models.py not found in current directory"
        exit 1
    fi
    
    # Make script executable
    chmod +x download_models.py
    
    # Run the downloader
    python3 download_models.py
    
    if [ $? -eq 0 ]; then
        print_success "Model download completed successfully!"
    else
        print_error "Model download failed!"
        exit 1
    fi
}

# Function to verify downloads
verify_downloads() {
    print_status "Verifying downloaded models..."
    
    # Check if models directory exists
    if [ ! -d "models" ]; then
        print_error "Models directory not found"
        exit 1
    fi
    
    # Check each model directory
    for model_dir in models/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            if [ "$(ls -A "$model_dir" 2>/dev/null)" ]; then
                print_success "$model_name: Verified"
            else
                print_warning "$model_name: Directory exists but is empty"
            fi
        fi
    done
}

# Function to show usage instructions
show_instructions() {
    echo
    print_status "Cloud GPU Deployment Instructions:"
    echo "=========================================="
    echo "1. Upload this script to your cloud GPU instance"
    echo "2. Make it executable: chmod +x download_models.sh"
    echo "3. Run: ./download_models.sh"
    echo "4. Models will be downloaded to /models directory"
    echo "5. Use generate_twin.py with the downloaded models"
    echo
    print_status "Tips for cloud deployment:"
    echo "• Ensure sufficient storage space (50GB+ recommended)"
    echo "• Use high-bandwidth connection for faster downloads"
    echo "• Consider using HuggingFace CLI for authentication"
    echo "• Models are cached for future use"
    echo
    print_status "Storage Requirements:"
    echo "• SDXL Base: ~6.9GB"
    echo "• IPAdapter: ~1.2GB"
    echo "• GFPGAN: ~340MB"
    echo "• InsightFace: ~500MB (auto-downloaded)"
    echo "• Total: ~9GB"
    echo
}

# Function to show help
show_help() {
    echo "Digital Twin Generator - Model Downloader"
    echo "========================================="
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --skip-hf      Skip HuggingFace authentication"
    echo "  --skip-deps    Skip dependency installation"
    echo "  --verify-only  Only verify existing downloads"
    echo
    echo "Environment Variables:"
    echo "  HF_TOKEN       HuggingFace token for authentication"
    echo "  CUDA_VISIBLE_DEVICES  GPU device selection"
    echo
    echo "Examples:"
    echo "  $0                    # Full download with authentication"
    echo "  $0 --skip-hf          # Download without HuggingFace login"
    echo "  $0 --verify-only      # Only verify existing models"
    echo
}

# Parse command line arguments
SKIP_HF=false
SKIP_DEPS=false
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-hf)
            SKIP_HF=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "🎯 Digital Twin Generator - Model Downloader"
    echo "============================================"
    echo "💡 Optimized for cloud GPU deployment"
    echo "💡 Models will be downloaded to /models directory"
    echo
    
    # Check requirements
    check_requirements
    
    # Detect environment
    detect_environment
    
    # Check storage
    check_storage
    
    # Optimize for cloud if needed
    optimize_for_cloud
    
    # Install dependencies if not skipped
    if [ "$SKIP_DEPS" = false ]; then
        install_dependencies
    fi
    
    # Setup HuggingFace if not skipped
    if [ "$SKIP_HF" = false ]; then
        setup_huggingface
    fi
    
    # Verify only mode
    if [ "$VERIFY_ONLY" = true ]; then
        verify_downloads
        exit 0
    fi
    
    # Run the downloader
    run_downloader
    
    # Verify downloads
    verify_downloads
    
    # Show instructions
    show_instructions
    
    print_success "🎉 Model download process completed!"
    print_success "🚀 Ready to generate digital twins!"
}

# Run main function
main "$@" 