/**
 * Premium Avatar Generator - JavaScript
 * Handles file upload, drag & drop, and real-time status updates
 */

class AvatarGenerator {
    constructor() {
        this.currentJobId = null;
        this.statusCheckInterval = null;
        this.uploadedFile = null;
        
        this.initializeElements();
        this.bindEvents();
        this.initializeDragAndDrop();
    }

    initializeElements() {
        // Core elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.generateBtn = document.getElementById('generateBtn');
        this.promptStyleSelect = document.getElementById('promptStyle');
        this.qualityModeSelect = document.getElementById('qualityMode');
        
        // Progress elements
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
        
        // File preview
        this.filePreview = document.getElementById('filePreview');
        this.fileInfo = document.getElementById('fileInfo');
    }

    bindEvents() {
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Generate button click
        this.generateBtn.addEventListener('click', () => {
            this.startGeneration();
        });

        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });

        // Download button click
        this.downloadBtn.addEventListener('click', () => {
            this.handleDownload();
        });

        // View toggle buttons
        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.handleViewToggle(e.target);
            });
        });

        // Slider handle
        if (this.sliderHandle) {
            this.sliderHandle.addEventListener('mousedown', (e) => {
                this.startSliderDrag(e);
            });
        }
    }

    initializeDragAndDrop() {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, () => {
                this.uploadArea.classList.remove('dragover');
            }, false);
        });

        // Handle dropped files
        this.uploadArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        }, false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!this.isValidFile(file)) {
            this.showStatus('Please upload a ZIP file containing your selfies.', 'error');
            return;
        }

        this.uploadedFile = file;
        this.showFilePreview(file);
        this.enableGenerateButton();
        this.showStatus('File selected successfully! Ready to generate avatar.', 'success');
    }

    isValidFile(file) {
        const allowedTypes = ['application/zip', 'application/x-zip-compressed'];
        const allowedExtensions = ['.zip'];
        
        // Check MIME type
        if (allowedTypes.includes(file.type)) {
            return true;
        }
        
        // Check file extension
        const fileName = file.name.toLowerCase();
        return allowedExtensions.some(ext => fileName.endsWith(ext));
    }

    showFilePreview(file) {
        const fileSize = this.formatFileSize(file.size);
        const fileName = file.name;
        
        this.fileInfo.innerHTML = `
            <span class="file-icon">📁</span>
            <span>${fileName}</span>
            <span>(${fileSize})</span>
        `;
        
        this.filePreview.classList.remove('hidden');
        this.filePreview.classList.add('fade-in');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    enableGenerateButton() {
        this.generateBtn.disabled = false;
        this.generateBtn.querySelector('.btn-content').innerHTML = `
            <span>🎨</span>
            <span>Generate Avatar</span>
        `;
    }

    disableGenerateButton() {
        this.generateBtn.disabled = true;
        this.generateBtn.querySelector('.btn-content').innerHTML = `
            <span class="spinner"></span>
            <span>Processing...</span>
        `;
    }

    async startGeneration() {
        if (!this.uploadedFile) {
            this.showStatus('Please select a file first.', 'error');
            return;
        }

        this.disableGenerateButton();
        this.hideResult();
        this.showProgress();

        try {
            const formData = new FormData();
            formData.append('file', this.uploadedFile);
            formData.append('prompt_style', this.promptStyleSelect.value);
            formData.append('quality_mode', this.qualityModeSelect.value);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.currentJobId = data.job_id;
                this.showStatus('Upload successful! Processing your selfies...', 'info');
                this.startProgressTracking();
            } else {
                this.showStatus(data.error || 'Upload failed. Please try again.', 'error');
                this.enableGenerateButton();
                this.hideProgress();
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showStatus('Network error. Please check your connection and try again.', 'error');
            this.enableGenerateButton();
            this.hideProgress();
        }
    }

    startProgressTracking() {
        this.updateProgress(10, 'Uploading files...');
        
        // Check status every 2 seconds
        this.statusCheckInterval = setInterval(() => {
            this.checkJobStatus();
        }, 2000);
    }

    async checkJobStatus() {
        if (!this.currentJobId) return;

        try {
            const response = await fetch(`/status/${this.currentJobId}`);
            const job = await response.json();

            if (response.ok) {
                this.updateProgress(job.progress, job.message);
                
                if (job.status === 'completed') {
                    this.handleGenerationComplete(job);
                } else if (job.status === 'failed') {
                    this.handleGenerationFailed(job.message);
                }
            } else {
                this.handleGenerationFailed('Failed to check job status');
            }
        } catch (error) {
            console.error('Status check error:', error);
            this.handleGenerationFailed('Network error during generation');
        }
    }

    handleGenerationComplete(job) {
        clearInterval(this.statusCheckInterval);
        this.updateProgress(100, 'Generation completed!');
        
        setTimeout(() => {
            this.hideProgress();
            this.showResult(job.result.images[0]);
            this.showStatus('Avatar generated successfully!', 'success');
            this.enableGenerateButton();
        }, 1000);
    }

    handleGenerationFailed(message) {
        clearInterval(this.statusCheckInterval);
        this.hideProgress();
        this.showStatus(message || 'Generation failed. Please try again.', 'error');
        this.enableGenerateButton();
    }

    updateProgress(progress, message) {
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = message || `Processing... ${progress}%`;
    }

    showProgress() {
        this.progressContainer.style.display = 'block';
        this.progressContainer.classList.add('fade-in');
    }

    hideProgress() {
        this.progressContainer.style.display = 'none';
    }

    showResult(imagePath) {
        const filename = imagePath.split('/').pop();
        
        // Show result container with animation
        this.resultContainer.style.display = 'block';
        this.resultContainer.classList.add('show');
        
        // Set the generated avatar image
        this.afterImage.src = `/download/${this.currentJobId}/${filename}`;
        this.afterImage.onload = () => {
            // Add fade-in animation
            this.comparisonSlider.classList.add('show');
        };
        
        // Set download link
        this.downloadBtn.setAttribute('data-download', `/download/${this.currentJobId}/${filename}`);
        this.downloadBtn.setAttribute('data-filename', `avatar_${filename}`);
        
        // Update quality info
        const qualityMode = this.qualityModeSelect.value;
        const qualityText = qualityMode === 'fast' ? 'Fast Mode' : 'High Fidelity Mode';
        this.qualityInfo.textContent = qualityText;
        
        // Populate selfie strip (demo data)
        this.populateSelfieStrip();
        
        // Initialize comparison slider
        this.initializeComparisonSlider();
    }

    hideResult() {
        this.resultContainer.style.display = 'none';
    }

    populateSelfieStrip() {
        // Clear existing selfies
        this.selfieStrip.innerHTML = '';
        
        // Add demo selfies (in a real app, these would come from the uploaded files)
        const demoSelfies = [
            '/static/images/demo-selfie-1.jpg',
            '/static/images/demo-selfie-2.jpg',
            '/static/images/demo-selfie-3.jpg',
            '/static/images/demo-selfie-4.jpg',
            '/static/images/demo-selfie-5.jpg'
        ];
        
        demoSelfies.forEach((src, index) => {
            const selfieItem = document.createElement('div');
            selfieItem.className = 'selfie-item';
            selfieItem.innerHTML = `
                <img src="${src}" alt="Selfie ${index + 1}" 
                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA4MCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjgwIiBoZWlnaHQ9IjgwIiBmaWxsPSIjRjNGNEY2Ii8+CjxjaXJjbGUgY3g9IjQwIiBjeT0iMzAiIHI9IjEwIiBmaWxsPSIjOUI5QkEwIi8+CjxwYXRoIGQ9Ik0yMCA1MEg2MFY2MEgyMFY1MFoiIGZpbGw9IiM5QjlCQTAiLz4KPC9zdmc+'">
            `;
            
            // Add click handler to select selfie
            selfieItem.addEventListener('click', () => {
                this.selectSelfie(selfieItem, src);
            });
            
            this.selfieStrip.appendChild(selfieItem);
        });
        
        // Add animation
        this.selfieStrip.classList.add('show');
    }

    selectSelfie(selfieItem, src) {
        // Remove previous selection
        document.querySelectorAll('.selfie-item').forEach(item => {
            item.classList.remove('selected');
        });
        
        // Select current item
        selfieItem.classList.add('selected');
        
        // Update before image in comparison
        this.beforeImage.src = src;
    }

    initializeComparisonSlider() {
        // Set initial before image
        const firstSelfie = this.selfieStrip.querySelector('.selfie-item img');
        if (firstSelfie) {
            this.beforeImage.src = firstSelfie.src;
        }
        
        // Initialize slider functionality
        this.setupSliderDrag();
    }

    setupSliderDrag() {
        let isDragging = false;
        let startX, startLeft;
        
        const handleMouseDown = (e) => {
            isDragging = true;
            startX = e.clientX;
            startLeft = this.sliderHandle.offsetLeft;
            document.body.style.cursor = 'ew-resize';
            e.preventDefault();
        };
        
        const handleMouseMove = (e) => {
            if (!isDragging) return;
            
            const deltaX = e.clientX - startX;
            const newLeft = startLeft + deltaX;
            const containerWidth = this.comparisonSlider.offsetWidth;
            const handleWidth = this.sliderHandle.offsetWidth;
            
            // Constrain to container bounds
            const constrainedLeft = Math.max(0, Math.min(newLeft, containerWidth - handleWidth));
            const percentage = (constrainedLeft / (containerWidth - handleWidth)) * 100;
            
            // Update handle position
            this.sliderHandle.style.left = `${constrainedLeft}px`;
            
            // Update clip path
            this.comparisonSlider.querySelector('.comparison-before').style.clipPath = 
                `polygon(0 0, ${percentage}% 0, ${percentage}% 100%, 0 100%)`;
        };
        
        const handleMouseUp = () => {
            isDragging = false;
            document.body.style.cursor = '';
        };
        
        this.sliderHandle.addEventListener('mousedown', handleMouseDown);
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    }

    handleViewToggle(button) {
        // Remove active class from all buttons
        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Update view
        const view = button.getAttribute('data-view');
        if (view === 'side-by-side') {
            this.comparisonSlider.classList.add('side-by-side');
        } else {
            this.comparisonSlider.classList.remove('side-by-side');
        }
    }

    async handleDownload() {
        const downloadUrl = this.downloadBtn.getAttribute('data-download');
        const filename = this.downloadBtn.getAttribute('data-filename');
        
        if (!downloadUrl) {
            this.showStatus('No avatar available for download', 'error');
            return;
        }
        
        // Show loading state
        this.downloadBtn.disabled = true;
        this.downloadBtn.querySelector('.btn-content').classList.add('hidden');
        this.downloadBtn.querySelector('.btn-loading').classList.remove('hidden');
        
        try {
            // Simulate download preparation
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Create download link
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showStatus('Avatar downloaded successfully!', 'success');
            
        } catch (error) {
            this.showStatus('Download failed. Please try again.', 'error');
        } finally {
            // Reset button state
            this.downloadBtn.disabled = false;
            this.downloadBtn.querySelector('.btn-content').classList.remove('hidden');
            this.downloadBtn.querySelector('.btn-loading').classList.add('hidden');
        }
    }

    showStatus(message, type = 'info') {
        // Remove existing status
        const existingStatus = this.statusContainer.querySelector('.status');
        if (existingStatus) {
            existingStatus.remove();
        }

        // Create new status
        const statusDiv = document.createElement('div');
        statusDiv.className = `status ${type}`;
        
        const icon = this.getStatusIcon(type);
        statusDiv.innerHTML = `
            <span>${icon}</span>
            <span>${message}</span>
        `;
        
        this.statusContainer.appendChild(statusDiv);
        statusDiv.classList.add('fade-in');
        
        // Auto-remove success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                if (statusDiv.parentNode) {
                    statusDiv.remove();
                }
            }, 5000);
        }
    }

    getStatusIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }

    // Utility methods
    reset() {
        this.currentJobId = null;
        this.uploadedFile = null;
        this.hideProgress();
        this.hideResult();
        this.filePreview.classList.add('hidden');
        this.enableGenerateButton();
        
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global instance
    window.avatarGenerator = new AvatarGenerator();
    
    // Add some nice loading animations
    document.body.classList.add('fade-in');
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to generate
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (!window.avatarGenerator.generateBtn.disabled) {
                window.avatarGenerator.startGeneration();
            }
        }
        
        // Escape to reset
        if (e.key === 'Escape') {
            window.avatarGenerator.reset();
        }
    });
    
    // Add some nice hover effects
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('mouseenter', () => {
        uploadArea.style.transform = 'translateY(-2px)';
    });
    
    uploadArea.addEventListener('mouseleave', () => {
        uploadArea.style.transform = 'translateY(0)';
    });
});

// Add some utility functions
window.utils = {
    // Format file size
    formatFileSize: (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // Debounce function
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Show loading overlay
    showLoading: (message = 'Processing...') => {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-modal">
                <div class="spinner" style="margin: 0 auto 1rem;"></div>
                <h3>Generating Avatar</h3>
                <p>${message}</p>
            </div>
        `;
        document.body.appendChild(overlay);
    },
    
    // Hide loading overlay
    hideLoading: () => {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
}; 