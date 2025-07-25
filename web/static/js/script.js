// Digital Twin Generator - Frontend JavaScript

class AvatarGenerator {
    constructor() {
        this.currentJobId = null;
        this.statusPollingInterval = null;
        this.avatarStyles = {};
        this.qualityModes = {};
        
        this.initializeElements();
        this.bindEvents();
        this.loadConfiguration();
    }

    initializeElements() {
        // Upload elements
        this.selfiesUploadArea = document.getElementById('selfiesUploadArea');
        this.selfiesFileInput = document.getElementById('selfiesFileInput');
        this.selfiesFilePreview = document.getElementById('selfiesFilePreview');
        this.selfiesFileInfo = document.getElementById('selfiesFileInfo');
        
        this.clothingUploadArea = document.getElementById('clothingUploadArea');
        this.clothingFileInput = document.getElementById('clothingFileInput');
        this.clothingFilePreview = document.getElementById('clothingFilePreview');
        this.clothingFileInfo = document.getElementById('clothingFileInfo');
        this.clothingPreview = document.getElementById('clothingPreview');
        
        // Customization elements
        this.customPrompt = document.getElementById('customPrompt');
        this.avatarStyle = document.getElementById('avatarStyle');
        this.styleDescription = document.getElementById('styleDescription');
        this.qualityModeGroup = document.getElementById('qualityModeGroup');
        
        // Action elements
        this.generateBtn = document.getElementById('generateBtn');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        
        // Result elements
        this.resultContainer = document.getElementById('resultContainer');
        this.selfieStrip = document.getElementById('selfieStrip');
        this.comparisonSlider = document.getElementById('comparisonSlider');
        this.sliderHandle = document.getElementById('sliderHandle');
        this.beforeImage = document.getElementById('beforeImage');
        this.afterImage = document.getElementById('afterImage');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.qualityInfo = document.getElementById('qualityInfo');
        
        // Status elements
        this.statusContainer = document.getElementById('statusContainer');
    }

    bindEvents() {
        // File upload events
        this.bindFileUploadEvents();
        
        // Customization events
        this.bindCustomizationEvents();
        
        // Generate button
        this.generateBtn.addEventListener('click', () => this.generateAvatar());
        
        // Download button
        this.downloadBtn.addEventListener('click', () => this.downloadAvatar());
        
        // Comparison slider
        this.bindComparisonSliderEvents();
        
        // View toggle
        this.bindViewToggleEvents();
    }

    bindFileUploadEvents() {
        // Selfies upload
        this.selfiesUploadArea.addEventListener('click', () => this.selfiesFileInput.click());
        this.selfiesUploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.selfiesUploadArea.addEventListener('drop', this.handleSelfiesDrop.bind(this));
        this.selfiesFileInput.addEventListener('change', this.handleSelfiesFileSelect.bind(this));
        
        // Clothing upload
        this.clothingUploadArea.addEventListener('click', () => this.clothingFileInput.click());
        this.clothingUploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.clothingUploadArea.addEventListener('drop', this.handleClothingDrop.bind(this));
        this.clothingFileInput.addEventListener('change', this.handleClothingFileSelect.bind(this));
    }

    bindCustomizationEvents() {
        // Avatar style change
        this.avatarStyle.addEventListener('change', () => this.updateStyleDescription());
        
        // Quality mode change
        this.qualityModeGroup.addEventListener('change', (e) => {
            if (e.target.name === 'qualityMode') {
                this.updateQualityInfo();
            }
        });
    }

    bindComparisonSliderEvents() {
        if (!this.sliderHandle) return;
        
        let isDragging = false;
        
        this.sliderHandle.addEventListener('mousedown', () => {
            isDragging = true;
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const rect = this.comparisonSlider.getBoundingClientRect();
            const x = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
            
            this.updateSliderPosition(x);
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        // Touch events for mobile
        this.sliderHandle.addEventListener('touchstart', () => {
            isDragging = true;
        });
        
        document.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            e.preventDefault();
            
            const touch = e.touches[0];
            const rect = this.comparisonSlider.getBoundingClientRect();
            const x = Math.max(0, Math.min(100, ((touch.clientX - rect.left) / rect.width) * 100));
            
            this.updateSliderPosition(x);
        });
        
        document.addEventListener('touchend', () => {
            isDragging = false;
        });
    }

    bindViewToggleEvents() {
        const toggleBtns = document.querySelectorAll('.toggle-btn');
        toggleBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active class from all buttons
                toggleBtns.forEach(b => b.classList.remove('active'));
                // Add active class to clicked button
                btn.classList.add('active');
                
                const view = btn.dataset.view;
                this.switchComparisonView(view);
            });
        });
    }

    async loadConfiguration() {
        try {
            // Load avatar styles
            const stylesResponse = await fetch('/avatar-styles');
            const stylesData = await stylesResponse.json();
            this.avatarStyles = stylesData.styles;
            
            // Load quality modes
            const modesResponse = await fetch('/quality-modes');
            const modesData = await modesResponse.json();
            this.qualityModes = modesData.modes;
            
            // Update initial descriptions
            this.updateStyleDescription();
            this.updateQualityInfo();
            
        } catch (error) {
            console.error('Failed to load configuration:', error);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleSelfiesDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.selfiesFileInput.files = files;
            this.handleSelfiesFileSelect();
        }
    }

    handleClothingDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.clothingFileInput.files = files;
            this.handleClothingFileSelect();
        }
    }

    handleSelfiesFileSelect() {
        const file = this.selfiesFileInput.files[0];
        if (file) {
            this.displaySelfiesFileInfo(file);
            this.validateForm();
        }
    }

    handleClothingFileSelect() {
        const file = this.clothingFileInput.files[0];
        if (file) {
            this.displayClothingFileInfo(file);
            this.validateForm();
        }
    }

    displaySelfiesFileInfo(file) {
        this.selfiesFileInfo.textContent = `${file.name} (${this.formatFileSize(file.size)})`;
        this.selfiesFilePreview.classList.remove('hidden');
    }

    displayClothingFileInfo(file) {
        this.clothingFileInfo.textContent = `${file.name} (${this.formatFileSize(file.size)})`;
        this.clothingFilePreview.classList.remove('hidden');
        
        // Preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            this.clothingPreview.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    updateStyleDescription() {
        const selectedStyle = this.avatarStyle.value;
        const styleConfig = this.avatarStyles[selectedStyle];
        
        if (styleConfig) {
            this.styleDescription.textContent = styleConfig.description;
        }
    }

    updateQualityInfo() {
        const selectedMode = document.querySelector('input[name="qualityMode"]:checked').value;
        const modeConfig = this.qualityModes[selectedMode];
        
        if (modeConfig) {
            this.qualityInfo.textContent = `${modeConfig.name} Mode`;
        }
    }

    validateForm() {
        const hasSelfies = this.selfiesFileInput.files.length > 0;
        this.generateBtn.disabled = !hasSelfies;
    }

    async generateAvatar() {
        if (!this.selfiesFileInput.files[0]) {
            this.showStatus('Please select a ZIP file with selfies', 'error');
            return;
        }

        const formData = new FormData();
        
        // Add ZIP file
        formData.append('zip_file', this.selfiesFileInput.files[0]);
        
        // Add clothing image if selected
        if (this.clothingFileInput.files[0]) {
            formData.append('clothing_image', this.clothingFileInput.files[0]);
        }
        
        // Add customization options
        formData.append('prompt', this.customPrompt.value.trim());
        formData.append('avatar_style', this.avatarStyle.value);
        formData.append('quality_mode', document.querySelector('input[name="qualityMode"]:checked').value);

        try {
            this.showProgress();
            this.generateBtn.disabled = true;
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.currentJobId = data.job_id;
                this.showStatus(`Upload successful! ${data.image_count} images found. Generation started.`, 'success');
                this.startStatusPolling();
            } else {
                this.showStatus(data.error || 'Upload failed', 'error');
                this.hideProgress();
                this.generateBtn.disabled = false;
            }

        } catch (error) {
            console.error('Generation error:', error);
            this.showStatus('Network error. Please try again.', 'error');
            this.hideProgress();
            this.generateBtn.disabled = false;
        }
    }

    showProgress() {
        this.progressContainer.classList.remove('hidden');
        this.progressFill.style.width = '0%';
        this.progressText.textContent = 'Preparing...';
    }

    hideProgress() {
        this.progressContainer.classList.add('hidden');
    }

    startStatusPolling() {
        if (this.statusPollingInterval) {
            clearInterval(this.statusPollingInterval);
        }

        this.statusPollingInterval = setInterval(async () => {
            if (!this.currentJobId) return;

            try {
                const response = await fetch(`/status/${this.currentJobId}`);
                const data = await response.json();

                if (response.ok) {
                    this.updateProgress(data.progress, data.message);
                    
                    if (data.status === 'completed') {
                        this.handleGenerationComplete(data);
                    } else if (data.status === 'failed') {
                        this.handleGenerationFailed(data.message);
                    }
                } else {
                    this.handleGenerationFailed('Failed to get job status');
                }

            } catch (error) {
                console.error('Status polling error:', error);
                this.handleGenerationFailed('Network error during generation');
            }
        }, 2000);
    }

    updateProgress(progress, message) {
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = message || 'Processing...';
    }

    handleGenerationComplete(data) {
        clearInterval(this.statusPollingInterval);
        this.statusPollingInterval = null;
        
        this.hideProgress();
        this.generateBtn.disabled = false;
        
        this.showStatus('Avatar generation completed!', 'success');
        this.displayResults(data);
    }

    handleGenerationFailed(message) {
        clearInterval(this.statusPollingInterval);
        this.statusPollingInterval = null;
        
        this.hideProgress();
        this.generateBtn.disabled = false;
        this.showStatus(message, 'error');
    }

    displayResults(data) {
        this.resultContainer.classList.remove('hidden');
        
        // Update quality info
        const qualityMode = data.quality_mode || 'high_fidelity';
        const modeConfig = this.qualityModes[qualityMode];
        if (modeConfig) {
            this.qualityInfo.textContent = `${modeConfig.name} Mode`;
        }
        
        // Set up download button
        if (data.result && data.result.output_path) {
            const filename = data.result.output_path.split('/').pop();
            this.downloadBtn.onclick = () => this.downloadFile(data.session_id, filename);
        }
        
        // Load comparison images
        this.loadComparisonImages(data);
        
        // Scroll to results
        this.resultContainer.scrollIntoView({ behavior: 'smooth' });
    }

    async loadComparisonImages(data) {
        // For now, we'll use placeholder images
        // In a real implementation, you'd load the actual generated images
        this.beforeImage.src = '/static/images/placeholder-before.jpg';
        this.afterImage.src = '/static/images/placeholder-after.jpg';
        
        // Load selfie strip (placeholder)
        this.loadSelfieStrip(data);
    }

    loadSelfieStrip(data) {
        // Clear existing selfies
        this.selfieStrip.innerHTML = '';
        
        // Add placeholder selfies (in real implementation, you'd load actual selfies)
        for (let i = 0; i < 5; i++) {
            const selfieItem = document.createElement('div');
            selfieItem.className = 'selfie-item';
            
            const img = document.createElement('img');
            img.src = `/static/images/placeholder-selfie-${i + 1}.jpg`;
            img.alt = `Selfie ${i + 1}`;
            
            selfieItem.appendChild(img);
            this.selfieStrip.appendChild(selfieItem);
        }
    }

    updateSliderPosition(percentage) {
        this.sliderHandle.style.left = `${percentage}%`;
        const afterElement = this.comparisonSlider.querySelector('.comparison-after');
        afterElement.style.width = `${percentage}%`;
    }

    switchComparisonView(view) {
        if (view === 'side-by-side') {
            this.comparisonSlider.classList.add('side-by-side');
            this.sliderHandle.style.display = 'none';
        } else {
            this.comparisonSlider.classList.remove('side-by-side');
            this.sliderHandle.style.display = 'block';
        }
    }

    async downloadFile(sessionId, filename) {
        try {
            this.downloadBtn.querySelector('.btn-content').classList.add('hidden');
            this.downloadBtn.querySelector('.btn-loading').classList.remove('hidden');
            
            const response = await fetch(`/download/${this.currentJobId}/${filename}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showStatus('Download completed!', 'success');
            } else {
                const error = await response.json();
                this.showStatus(error.error || 'Download failed', 'error');
            }
            
        } catch (error) {
            console.error('Download error:', error);
            this.showStatus('Download failed', 'error');
        } finally {
            this.downloadBtn.querySelector('.btn-content').classList.remove('hidden');
            this.downloadBtn.querySelector('.btn-loading').classList.add('hidden');
        }
    }

    showStatus(message, type = 'info') {
        const statusDiv = document.createElement('div');
        statusDiv.className = `status-message ${type}`;
        statusDiv.textContent = message;
        
        this.statusContainer.appendChild(statusDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (statusDiv.parentNode) {
                statusDiv.parentNode.removeChild(statusDiv);
            }
        }, 5000);
    }

    // Utility methods
    downloadAvatar() {
        // This would trigger the download of the generated avatar
        if (this.currentJobId) {
            this.downloadFile(this.currentJobId, 'avatar.png');
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AvatarGenerator();
});

// Add some utility functions for better UX
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add smooth scrolling for better UX
function smoothScrollTo(element) {
    element.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const generateBtn = document.getElementById('generateBtn');
        if (!generateBtn.disabled) {
            generateBtn.click();
        }
    }
}); 