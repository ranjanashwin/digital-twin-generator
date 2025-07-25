/**
 * Fashion Digital Twin Generator - Workflow JavaScript
 */

class FashionWorkflow {
    constructor() {
        this.currentStep = 1;
        this.sessionId = null;
        this.currentJobId = null;
        this.pollingInterval = null;
        
        this.initializeElements();
        this.bindEvents();
        this.loadConfiguration();
    }
    
    initializeElements() {
        // Step elements
        this.step1 = document.getElementById('step1');
        this.step2 = document.getElementById('step2');
        this.step3 = document.getElementById('step3');
        
        // Upload areas
        this.selfiesUploadArea = document.getElementById('selfiesUploadArea');
        this.clothingUploadArea = document.getElementById('clothingUploadArea');
        
        // File inputs
        this.selfiesFileInput = document.getElementById('selfiesFileInput');
        this.clothingFileInput = document.getElementById('clothingFileInput');
        
        // File previews
        this.selfiesFilePreview = document.getElementById('selfiesFilePreview');
        this.clothingFilePreview = document.getElementById('clothingFilePreview');
        
        // Customization elements
        this.avatarStyle = document.getElementById('avatarStyle');
        this.customPrompt = document.getElementById('customPrompt');
        this.scenePrompt = document.getElementById('scenePrompt');
        this.styleDescription = document.getElementById('styleDescription');
        
        // Quality mode groups
        this.qualityModeGroup = document.getElementById('qualityModeGroup');
        this.fashionQualityModeGroup = document.getElementById('fashionQualityModeGroup');
        
        // Buttons
        this.generateAvatarBtn = document.getElementById('generateAvatarBtn');
        this.generateFashionBtn = document.getElementById('generateFashionBtn');
        this.downloadAvatarBtn = document.getElementById('downloadAvatarBtn');
        this.downloadFashionBtn = document.getElementById('downloadFashionBtn');
        this.regenerateBtn = document.getElementById('regenerateBtn');
        this.restartBtn = document.getElementById('restartBtn');
        
        // Progress elements
        this.progressContainer = document.getElementById('progressContainer');
        this.progressTitle = document.getElementById('progressTitle');
        this.progressFill = document.getElementById('progressFill');
        this.progressMessage = document.getElementById('progressMessage');
        
        // Status elements
        this.statusMessage = document.getElementById('statusMessage');
        this.resultsContainer = document.getElementById('resultsContainer');
        
        // Step status elements
        this.step1Status = document.getElementById('step1-status');
        this.step2Status = document.getElementById('step2-status');
        this.step3Status = document.getElementById('step3-status');
    }
    
    bindEvents() {
        // File upload events
        this.bindFileUploadEvents();
        
        // Customization events
        this.bindCustomizationEvents();
        
        // Button events
        this.bindButtonEvents();
        
        // Form validation
        this.bindValidationEvents();
    }
    
    bindFileUploadEvents() {
        // Selfies upload
        this.selfiesUploadArea.addEventListener('click', () => this.selfiesFileInput.click());
        this.selfiesFileInput.addEventListener('change', (e) => this.handleSelfiesFileSelect(e));
        this.initializeDragAndDrop(this.selfiesUploadArea, this.handleSelfiesDrop.bind(this));
        
        // Clothing upload
        this.clothingUploadArea.addEventListener('click', () => this.clothingFileInput.click());
        this.clothingFileInput.addEventListener('change', (e) => this.handleClothingFileSelect(e));
        this.initializeDragAndDrop(this.clothingUploadArea, this.handleClothingDrop.bind(this));
    }
    
    bindCustomizationEvents() {
        // Avatar style change
        this.avatarStyle.addEventListener('change', () => this.updateStyleDescription());
        
        // Quality mode changes
        this.qualityModeGroup.addEventListener('change', () => this.updateQualityInfo());
        this.fashionQualityModeGroup.addEventListener('change', () => this.updateQualityInfo());
    }
    
    bindButtonEvents() {
        // Generate buttons
        this.generateAvatarBtn.addEventListener('click', () => this.generateAvatar());
        this.generateFashionBtn.addEventListener('click', () => this.generateFashionPhoto());
        
        // Download buttons
        this.downloadAvatarBtn.addEventListener('click', () => this.downloadFile('avatar'));
        this.downloadFashionBtn.addEventListener('click', () => this.downloadFile('fashion'));
        
        // Action buttons
        this.regenerateBtn.addEventListener('click', () => this.regeneratePhoto());
        this.restartBtn.addEventListener('click', () => this.restartWorkflow());
    }
    
    bindValidationEvents() {
        // Real-time validation
        this.selfiesFileInput.addEventListener('change', () => this.validateForm());
        this.clothingFileInput.addEventListener('change', () => this.validateForm());
        this.scenePrompt.addEventListener('input', () => this.validateForm());
    }
    
    initializeDragAndDrop(dropZone, onDrop) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                onDrop(files[0]);
            }
        });
    }
    
    handleSelfiesDrop(file) {
        if (file.type === 'application/zip' || file.name.endsWith('.zip')) {
            this.selfiesFileInput.files = new FileList([file]);
            this.handleSelfiesFileSelect({ target: { files: [file] } });
        } else {
            this.showStatus('Please upload a ZIP file containing selfies', 'error');
        }
    }
    
    handleClothingDrop(file) {
        if (file.type.startsWith('image/')) {
            this.clothingFileInput.files = new FileList([file]);
            this.handleClothingFileSelect({ target: { files: [file] } });
        } else {
            this.showStatus('Please upload an image file', 'error');
        }
    }
    
    handleSelfiesFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.displaySelfiesFileInfo(file);
            this.validateForm();
        }
    }
    
    handleClothingFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.displayClothingFileInfo(file);
            this.validateForm();
        }
    }
    
    displaySelfiesFileInfo(file) {
        this.selfiesFilePreview.style.display = 'block';
        this.selfiesFileName.textContent = file.name;
        this.selfiesFileSize.textContent = this.formatFileSize(file.size);
        
        this.selfiesUploadArea.style.display = 'none';
    }
    
    displayClothingFileInfo(file) {
        this.clothingFilePreview.style.display = 'block';
        this.clothingFileName.textContent = file.name;
        this.clothingFileSize.textContent = this.formatFileSize(file.size);
        
        // Create preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            this.clothingPreviewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        this.clothingUploadArea.style.display = 'none';
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    loadConfiguration() {
        // Load fashion styles
        fetch('/fashion-styles')
            .then(response => response.json())
            .then(data => {
                this.fashionStyles = data.styles;
                this.updateStyleDescription();
            })
            .catch(error => {
                console.error('Failed to load fashion styles:', error);
            });
        
        // Load quality modes
        fetch('/fashion-quality-modes')
            .then(response => response.json())
            .then(data => {
                this.qualityModes = data.modes;
                this.updateQualityInfo();
            })
            .catch(error => {
                console.error('Failed to load quality modes:', error);
            });
    }
    
    updateStyleDescription() {
        const selectedStyle = this.avatarStyle.value;
        const styleConfig = this.fashionStyles?.[selectedStyle];
        
        if (styleConfig) {
            this.styleDescription.textContent = styleConfig.description;
        }
    }
    
    updateQualityInfo() {
        // Update quality mode information if needed
        const selectedMode = this.qualityModeGroup.querySelector('input:checked')?.value;
        if (selectedMode && this.qualityModes?.[selectedMode]) {
            // Update any quality-specific UI elements
        }
    }
    
    validateForm() {
        let isValid = false;
        
        if (this.currentStep === 1) {
            // Validate step 1: selfies upload
            isValid = this.selfiesFileInput.files.length > 0;
            this.generateAvatarBtn.disabled = !isValid;
        } else if (this.currentStep === 2) {
            // Validate step 2: clothing and scene
            const hasClothing = this.clothingFileInput.files.length > 0;
            const hasScenePrompt = this.scenePrompt.value.trim().length > 0;
            isValid = hasClothing && hasScenePrompt;
            this.generateFashionBtn.disabled = !isValid;
        }
    }
    
    generateAvatar() {
        if (!this.selfiesFileInput.files[0]) {
            this.showStatus('Please upload a ZIP file with selfies', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('zip_file', this.selfiesFileInput.files[0]);
        formData.append('prompt', this.customPrompt.value.trim());
        formData.append('avatar_style', this.avatarStyle.value);
        formData.append('quality_mode', this.qualityModeGroup.querySelector('input:checked').value);
        
        this.startGeneration('/upload-selfies', formData, 'Avatar generation started...');
    }
    
    generateFashionPhoto() {
        if (!this.sessionId) {
            this.showStatus('Please complete step 1 first', 'error');
            return;
        }
        
        if (!this.clothingFileInput.files[0]) {
            this.showStatus('Please upload a clothing image', 'error');
            return;
        }
        
        if (!this.scenePrompt.value.trim()) {
            this.showStatus('Please provide a scene description', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('session_id', this.sessionId);
        formData.append('clothing_image', this.clothingFileInput.files[0]);
        formData.append('scene_prompt', this.scenePrompt.value.trim());
        formData.append('quality_mode', this.fashionQualityModeGroup.querySelector('input:checked').value);
        
        this.startGeneration('/upload-clothing-scene', formData, 'Fashion photo generation started...');
    }
    
    startGeneration(endpoint, formData, message) {
        this.showProgress(message);
        this.updateStepStatus(this.currentStep, 'processing');
        
        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.currentJobId = data.job_id;
            this.sessionId = data.session_id;
            
            this.showStatus(data.message, 'success');
            this.startStatusPolling();
        })
        .catch(error => {
            this.showStatus(`Generation failed: ${error.message}`, 'error');
            this.updateStepStatus(this.currentStep, 'failed');
            this.hideProgress();
        });
    }
    
    startStatusPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        this.pollingInterval = setInterval(() => {
            this.checkJobStatus();
        }, 2000);
    }
    
    checkJobStatus() {
        if (!this.currentJobId) return;
        
        fetch(`/status/${this.currentJobId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                this.updateProgress(data.progress, data.message);
                
                if (data.status === 'completed') {
                    this.handleGenerationComplete(data);
                } else if (data.status === 'failed') {
                    this.handleGenerationFailed(data);
                }
            })
            .catch(error => {
                console.error('Status check failed:', error);
                this.handleGenerationFailed({ message: error.message });
            });
    }
    
    handleGenerationComplete(data) {
        clearInterval(this.pollingInterval);
        this.pollingInterval = null;
        
        this.updateStepStatus(this.currentStep, 'completed');
        this.hideProgress();
        
        if (this.currentStep === 1) {
            // Avatar generation completed
            this.showStatus('Avatar generation completed!', 'success');
            this.advanceToStep(2);
        } else if (this.currentStep === 2) {
            // Fashion photo generation completed
            this.showStatus('Fashion photo generation completed!', 'success');
            this.advanceToStep(3);
            this.displayResults(data);
        }
    }
    
    handleGenerationFailed(data) {
        clearInterval(this.pollingInterval);
        this.pollingInterval = null;
        
        this.updateStepStatus(this.currentStep, 'failed');
        this.hideProgress();
        this.showStatus(`Generation failed: ${data.message}`, 'error');
    }
    
    advanceToStep(step) {
        this.currentStep = step;
        
        // Hide current step
        document.getElementById(`step${step - 1}`).style.display = 'none';
        
        // Show next step
        document.getElementById(`step${step}`).style.display = 'block';
        
        // Update step status
        this.updateStepStatus(step, 'pending');
    }
    
    updateStepStatus(step, status) {
        const statusElement = document.getElementById(`step${step}-status`);
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `step-status ${status}`;
        }
    }
    
    displayResults(data) {
        this.resultsContainer.style.display = 'block';
        
        // Update result images (placeholder for now)
        // In a real implementation, you would load the actual generated images
        this.avatarResult.src = '/static/images/placeholder.png';
        this.fashionResult.src = '/static/images/placeholder.png';
    }
    
    downloadFile(type) {
        if (!this.currentJobId) {
            this.showStatus('No file available for download', 'error');
            return;
        }
        
        // Determine filename based on type
        let filename = '';
        if (type === 'avatar') {
            filename = 'avatar_001.png';
        } else if (type === 'fashion') {
            filename = 'fashion_photo_001.png';
        }
        
        if (filename) {
            const downloadUrl = `/download/${this.currentJobId}/${filename}`;
            window.open(downloadUrl, '_blank');
        }
    }
    
    regeneratePhoto() {
        // Go back to step 2 to regenerate fashion photo
        this.currentStep = 2;
        this.step1.style.display = 'none';
        this.step2.style.display = 'block';
        this.step3.style.display = 'none';
        this.resultsContainer.style.display = 'none';
    }
    
    restartWorkflow() {
        // Reset everything and go back to step 1
        this.currentStep = 1;
        this.sessionId = null;
        this.currentJobId = null;
        
        // Reset file inputs
        this.selfiesFileInput.value = '';
        this.clothingFileInput.value = '';
        
        // Reset previews
        this.selfiesFilePreview.style.display = 'none';
        this.clothingFilePreview.style.display = 'none';
        this.selfiesUploadArea.style.display = 'block';
        this.clothingUploadArea.style.display = 'block';
        
        // Reset forms
        this.customPrompt.value = '';
        this.scenePrompt.value = '';
        
        // Reset step statuses
        this.updateStepStatus(1, 'pending');
        this.updateStepStatus(2, 'pending');
        this.updateStepStatus(3, 'pending');
        
        // Show step 1
        this.step1.style.display = 'block';
        this.step2.style.display = 'none';
        this.step3.style.display = 'none';
        
        // Validate form
        this.validateForm();
        
        this.showStatus('Workflow restarted', 'success');
    }
    
    showProgress(message) {
        this.progressContainer.style.display = 'block';
        this.progressTitle.textContent = 'Processing...';
        this.progressMessage.textContent = message;
        this.progressFill.style.width = '0%';
    }
    
    updateProgress(progress, message) {
        this.progressFill.style.width = `${progress}%`;
        this.progressMessage.textContent = message;
    }
    
    hideProgress() {
        this.progressContainer.style.display = 'none';
    }
    
    showStatus(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
        this.statusMessage.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.statusMessage.style.display = 'none';
        }, 5000);
    }
    
    smoothScrollTo(element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Utility function for FileList (not available in all browsers)
class FileList {
    constructor(files) {
        this.files = files;
        this.length = files.length;
    }
    
    item(index) {
        return this.files[index];
    }
    
    [Symbol.iterator]() {
        return this.files[Symbol.iterator]();
    }
}

// Debounce utility
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

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.fashionWorkflow = new FashionWorkflow();
}); 