/**
 * Fashion Digital Twin Generator - Workflow JavaScript
 */

class FashionWorkflow {
    constructor() {
        this.currentStep = 1;
        this.sessionId = null;
        this.avatarJobId = null;
        this.fashionJobId = null;
        this.clothingItems = [];
        this.posePresets = {};
        this.clothingTypes = {};
        
        this.initializeElements();
        this.bindEvents();
        this.loadConfiguration();
    }
    
    initializeElements() {
        // Step elements
        this.step1 = document.getElementById('step1');
        this.step2 = document.getElementById('step2');
        this.step3 = document.getElementById('step3');
        
        // Step status elements
        this.step1Status = document.getElementById('step1-status');
        this.step2Status = document.getElementById('step2-status');
        this.step3Status = document.getElementById('step3-status');
        
        // File upload elements
        this.selfiesUploadArea = document.getElementById('selfiesUploadArea');
        this.selfiesFileInput = document.getElementById('selfiesFileInput');
        this.selfiesFilePreview = document.getElementById('selfiesFilePreview');
        this.selfiesFileName = document.getElementById('selfiesFileName');
        this.selfiesFileSize = document.getElementById('selfiesFileSize');
        
        this.clothingUploadArea = document.getElementById('clothingUploadArea');
        this.clothingFileInput = document.getElementById('clothingFileInput');
        this.clothingPreviewContainer = document.getElementById('clothingPreviewContainer');
        this.clothingItemsGrid = document.getElementById('clothingItemsGrid');
        
        // Pose control elements
        this.poseControlType = document.getElementById('poseControlType');
        this.referencePoseContent = document.getElementById('referencePoseContent');
        this.presetPoseContent = document.getElementById('presetPoseContent');
        this.referencePoseUploadArea = document.getElementById('referencePoseUploadArea');
        this.referencePoseFileInput = document.getElementById('referencePoseFileInput');
        this.referencePoseFilePreview = document.getElementById('referencePoseFilePreview');
        this.referencePosePreviewImage = document.getElementById('referencePosePreviewImage');
        this.referencePoseFileName = document.getElementById('referencePoseFileName');
        this.referencePoseFileSize = document.getElementById('referencePoseFileSize');
        this.posePreset = document.getElementById('posePreset');
        this.presetDescription = document.getElementById('presetDescription');
        
        // Compatibility elements
        this.compatibilitySection = document.getElementById('compatibilitySection');
        this.compatibilityStatus = document.getElementById('compatibilityStatus');
        this.compatibilitySuggestions = document.getElementById('compatibilitySuggestions');
        
        // Form elements
        this.avatarStyle = document.getElementById('avatarStyle');
        this.customPrompt = document.getElementById('customPrompt');
        this.scenePrompt = document.getElementById('scenePrompt');
        this.styleDescription = document.getElementById('styleDescription');
        
        // Quality mode elements
        this.qualityModeGroup = document.getElementById('qualityModeGroup');
        this.fashionQualityModeGroup = document.getElementById('fashionQualityModeGroup');
        
        // Button elements
        this.generateAvatarBtn = document.getElementById('generateAvatarBtn');
        this.generateFashionBtn = document.getElementById('generateFashionBtn');
        
        // Progress elements
        this.progressContainer = document.getElementById('progressContainer');
        this.progressTitle = document.getElementById('progressTitle');
        this.progressFill = document.getElementById('progressFill');
        this.progressMessage = document.getElementById('progressMessage');
        
        // Status elements
        this.statusMessage = document.getElementById('statusMessage');
        
        // Results elements
        this.resultsContainer = document.getElementById('resultsContainer');
        this.avatarResult = document.getElementById('avatarResult');
        this.fashionResult = document.getElementById('fashionResult');
        this.downloadAvatarBtn = document.getElementById('downloadAvatarBtn');
        this.downloadFashionBtn = document.getElementById('downloadFashionBtn');
        this.regenerateBtn = document.getElementById('regenerateBtn');
        this.restartBtn = document.getElementById('restartBtn');
    }
    
    bindEvents() {
        // File upload events
        this.initializeDragAndDrop();
        this.selfiesFileInput.addEventListener('change', (e) => this.handleSelfiesDrop(e.target.files));
        this.clothingFileInput.addEventListener('change', (e) => this.handleClothingDrop(e.target.files));
        this.referencePoseFileInput.addEventListener('change', (e) => this.handleReferencePoseDrop(e.target.files));
        
        // Pose control events
        this.poseControlType.addEventListener('change', () => this.handlePoseControlChange());
        this.posePreset.addEventListener('change', () => this.handlePosePresetChange());
        
        // Form validation events
        this.avatarStyle.addEventListener('change', () => this.updateStyleDescription());
        this.customPrompt.addEventListener('input', () => this.validateForm());
        this.scenePrompt.addEventListener('input', () => this.validateForm());
        
        // Button events
        this.generateAvatarBtn.addEventListener('click', () => this.generateAvatar());
        this.generateFashionBtn.addEventListener('click', () => this.generateFashionPhoto());
        this.downloadAvatarBtn.addEventListener('click', () => this.downloadFile('avatar'));
        this.downloadFashionBtn.addEventListener('click', () => this.downloadFile('fashion'));
        this.regenerateBtn.addEventListener('click', () => this.regeneratePhoto());
        this.restartBtn.addEventListener('click', () => this.restartWorkflow());
        
        // Quality mode events
        this.qualityModeGroup.addEventListener('change', () => this.updateQualityInfo());
        this.fashionQualityModeGroup.addEventListener('change', () => this.updateQualityInfo());
    }
    
    initializeDragAndDrop() {
        // Selfies drag and drop
        this.selfiesUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.selfiesUploadArea.classList.add('dragover');
        });
        
        this.selfiesUploadArea.addEventListener('dragleave', () => {
            this.selfiesUploadArea.classList.remove('dragover');
        });
        
        this.selfiesUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.selfiesUploadArea.classList.remove('dragover');
            this.handleSelfiesDrop(e.dataTransfer.files);
        });
        
        this.selfiesUploadArea.addEventListener('click', () => {
            this.selfiesFileInput.click();
        });
        
        // Clothing drag and drop
        this.clothingUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.clothingUploadArea.classList.add('dragover');
        });
        
        this.clothingUploadArea.addEventListener('dragleave', () => {
            this.clothingUploadArea.classList.remove('dragover');
        });
        
        this.clothingUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.clothingUploadArea.classList.remove('dragover');
            this.handleClothingDrop(e.dataTransfer.files);
        });
        
        this.clothingUploadArea.addEventListener('click', () => {
            this.clothingFileInput.click();
        });
        
        // Reference pose drag and drop
        this.referencePoseUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.referencePoseUploadArea.classList.add('dragover');
        });
        
        this.referencePoseUploadArea.addEventListener('dragleave', () => {
            this.referencePoseUploadArea.classList.remove('dragover');
        });
        
        this.referencePoseUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.referencePoseUploadArea.classList.remove('dragover');
            this.handleReferencePoseDrop(e.dataTransfer.files);
        });
        
        this.referencePoseUploadArea.addEventListener('click', () => {
            this.referencePoseFileInput.click();
        });
    }
    
    handleSelfiesDrop(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Check file size (5GB limit)
        const maxSize = 5 * 1024 * 1024 * 1024; // 5GB in bytes
        if (file.size > maxSize) {
            this.showStatus('File size too large. Maximum size is 5GB.', 'error');
            return;
        }
        
        // Check file type
        if (!file.name.toLowerCase().endsWith('.zip')) {
            this.showStatus('Please upload a ZIP file containing selfies', 'error');
            return;
        }
        
        // Show upload progress
        this.showUploadProgress('Uploading selfies...');
        
        this.displaySelfiesFileInfo(file);
        this.validateForm();
    }
    
    handleClothingDrop(files) {
        if (files.length === 0) return;
        
        // Check total file size
        const maxSize = 5 * 1024 * 1024 * 1024; // 5GB in bytes
        const totalSize = Array.from(files).reduce((sum, file) => sum + file.size, 0);
        
        if (totalSize > maxSize) {
            this.showStatus('Total file size too large. Maximum size is 5GB.', 'error');
            return;
        }
        
        // Clear existing items
        this.clothingItems = [];
        this.clothingItemsGrid.innerHTML = '';
        
        // Process each file
        let validFiles = 0;
        Array.from(files).forEach((file, index) => {
            // Check individual file size
            if (file.size > maxSize) {
                this.showStatus(`File ${file.name} is too large. Maximum size is 5GB.`, 'error');
                return;
            }
            
            // Check file type
            const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif'];
            if (!allowedTypes.includes(file.type)) {
                this.showStatus(`File ${file.name} is not a supported image type.`, 'error');
                return;
            }
            
            const clothingItem = {
                file: file,
                name: file.name.replace(/\.[^/.]+$/, ''),
                type: 'top',
                layer: 2
            };
            
            this.clothingItems.push(clothingItem);
            this.displayClothingItem(clothingItem, index);
            validFiles++;
        });
        
        if (validFiles > 0) {
            this.clothingPreviewContainer.style.display = 'block';
            this.validateForm();
            this.checkClothingCompatibility();
            this.showStatus(`Successfully added ${validFiles} clothing item(s)`, 'success');
        }
    }
    
    handleReferencePoseDrop(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Check file size
        const maxSize = 100 * 1024 * 1024; // 100MB limit for reference pose
        if (file.size > maxSize) {
            this.showStatus('Reference pose file too large. Maximum size is 100MB.', 'error');
            return;
        }
        
        // Check file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            this.showStatus('Please upload an image file for reference pose (PNG, JPEG, WebP)', 'error');
            return;
        }
        
        this.displayReferencePoseFileInfo(file);
        this.showStatus('Reference pose uploaded successfully', 'success');
    }
    
    displaySelfiesFileInfo(file) {
        this.selfiesFileName.textContent = file.name;
        this.selfiesFileSize.textContent = this.formatFileSize(file.size);
        this.selfiesFilePreview.style.display = 'block';
    }
    
    displayClothingItem(clothingItem, index) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const card = document.createElement('div');
            card.className = 'clothing-item-card';
            card.innerHTML = `
                <img src="${e.target.result}" alt="${clothingItem.name}" class="clothing-item-image">
                <div class="clothing-item-info">
                    <div class="clothing-item-name">${clothingItem.name}</div>
                    <div class="clothing-item-type">${this.getClothingTypeLabel(clothingItem.type)}</div>
                    <div class="clothing-item-layer">${this.getClothingLayerLabel(clothingItem.layer)}</div>
                    <div class="clothing-item-actions">
                        <button class="edit-btn" onclick="fashionWorkflow.editClothingItem(${index})">Edit</button>
                        <button class="remove-btn" onclick="fashionWorkflow.removeClothingItem(${index})">Remove</button>
                    </div>
                </div>
            `;
            this.clothingItemsGrid.appendChild(card);
        };
        reader.readAsDataURL(clothingItem.file);
    }
    
    displayReferencePoseFileInfo(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.referencePosePreviewImage.src = e.target.result;
            this.referencePoseFileName.textContent = file.name;
            this.referencePoseFileSize.textContent = this.formatFileSize(file.size);
            this.referencePoseFilePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
    
    editClothingItem(index) {
        const item = this.clothingItems[index];
        const newName = prompt('Enter clothing item name:', item.name);
        if (newName) {
            item.name = newName;
            this.refreshClothingItems();
        }
    }
    
    removeClothingItem(index) {
        this.clothingItems.splice(index, 1);
        this.refreshClothingItems();
        this.validateForm();
        this.checkClothingCompatibility();
    }
    
    refreshClothingItems() {
        this.clothingItemsGrid.innerHTML = '';
        this.clothingItems.forEach((item, index) => {
            this.displayClothingItem(item, index);
        });
    }
    
    handlePoseControlChange() {
        const controlType = this.poseControlType.value;
        
        // Hide all pose option content
        this.referencePoseContent.style.display = 'none';
        this.presetPoseContent.style.display = 'none';
        
        // Show relevant content
        if (controlType === 'reference') {
            this.referencePoseContent.style.display = 'block';
        } else if (controlType === 'preset') {
            this.presetPoseContent.style.display = 'block';
        }
    }
    
    handlePosePresetChange() {
        const presetName = this.posePreset.value;
        if (presetName && this.posePresets[presetName]) {
            const preset = this.posePresets[presetName];
            this.presetDescription.textContent = preset.description;
            this.presetDescription.style.display = 'block';
        } else {
            this.presetDescription.style.display = 'none';
        }
    }
    
    async loadConfiguration() {
        try {
            // Load pose presets
            const poseResponse = await fetch('/api/pose-presets');
            if (poseResponse.ok) {
                const poseData = await poseResponse.json();
                this.posePresets = poseData.presets;
                this.populatePosePresets();
            }
            
            // Load clothing types
            const clothingResponse = await fetch('/api/clothing-types');
            if (clothingResponse.ok) {
                const clothingData = await clothingResponse.json();
                this.clothingTypes = clothingData.clothing_types;
            }
            
        } catch (error) {
            console.error('Error loading configuration:', error);
        }
    }
    
    populatePosePresets() {
        this.posePreset.innerHTML = '<option value="">Select a pose preset...</option>';
        
        Object.entries(this.posePresets).forEach(([key, preset]) => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = preset.description || key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            this.posePreset.appendChild(option);
        });
    }
    
    getClothingTypeLabel(type) {
        const typeMap = {
            'top': 'Top',
            'bottom': 'Bottom',
            'dress': 'Dress',
            'outerwear': 'Outerwear',
            'accessories': 'Accessories',
            'shoes': 'Shoes'
        };
        return typeMap[type] || type;
    }
    
    getClothingLayerLabel(layer) {
        const layerMap = {
            0: 'Underwear',
            1: 'Bottom',
            2: 'Top',
            3: 'Outerwear',
            4: 'Accessories'
        };
        return layerMap[layer] || `Layer ${layer}`;
    }
    
    async checkClothingCompatibility() {
        if (this.clothingItems.length === 0) {
            this.compatibilitySection.style.display = 'none';
            return;
        }
        
        try {
            const formData = new FormData();
            this.clothingItems.forEach((item, index) => {
                formData.append(`clothing_items[${index}][image_path]`, item.file);
                formData.append(`clothing_items[${index}][type]`, item.type);
                formData.append(`clothing_items[${index}][layer]`, item.layer);
                formData.append(`clothing_items[${index}][name]`, item.name);
            });
            
            const response = await fetch('/api/validate-clothing', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.displayCompatibilityResult(result);
            }
        } catch (error) {
            console.error('Error checking compatibility:', error);
        }
    }
    
    displayCompatibilityResult(result) {
        this.compatibilitySection.style.display = 'block';
        
        if (result.compatible) {
            this.compatibilityStatus.className = 'compatibility-status compatible';
            this.compatibilityStatus.textContent = '✅ All clothing items are compatible';
        } else {
            this.compatibilityStatus.className = 'compatibility-status incompatible';
            this.compatibilityStatus.textContent = '❌ Clothing compatibility issues detected';
        }
        
        if (result.suggestions && result.suggestions.length > 0) {
            this.compatibilitySuggestions.innerHTML = `
                <h5>Suggestions:</h5>
                <ul>
                    ${result.suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
                </ul>
            `;
        } else {
            this.compatibilitySuggestions.innerHTML = '';
        }
    }
    
    updateStyleDescription() {
        const style = this.avatarStyle.value;
        const descriptions = {
            'fashion_portrait': 'Professional fashion portrait with clean background',
            'street_style': 'Casual street style with urban environment',
            'studio_fashion': 'High-end studio fashion with professional lighting',
            'editorial': 'Dramatic editorial style with artistic composition'
        };
        this.styleDescription.textContent = descriptions[style] || '';
    }
    
    updateQualityInfo() {
        // Update quality mode information
        const qualityModes = {
            'standard': 'Good quality, faster generation (3-5 min)',
            'high_fidelity': 'Excellent quality, balanced speed (5-8 min)',
            'ultra_fidelity': 'Maximum quality, slower generation (8-12 min)'
        };
        
        // Update both quality mode groups
        [this.qualityModeGroup, this.fashionQualityModeGroup].forEach(group => {
            const selected = group.querySelector('input:checked');
            if (selected) {
                const description = selected.closest('.radio-option').querySelector('.radio-description');
                description.textContent = qualityModes[selected.value] || '';
            }
        });
    }
    
    validateForm() {
        let isValid = true;
        
        // Step 1 validation
        if (this.currentStep === 1) {
            const hasSelfies = this.selfiesFilePreview.style.display !== 'none';
            isValid = isValid && hasSelfies;
        }
        
        // Step 2 validation
        if (this.currentStep === 2) {
            const hasClothing = this.clothingItems.length > 0;
            const hasScenePrompt = this.scenePrompt.value.trim().length > 0;
            isValid = isValid && hasClothing && hasScenePrompt;
        }
        
        // Update button states
        if (this.currentStep === 1) {
            this.generateAvatarBtn.disabled = !isValid;
        } else if (this.currentStep === 2) {
            this.generateFashionBtn.disabled = !isValid;
        }
        
        return isValid;
    }
    
    async generateAvatar() {
        if (!this.validateForm()) return;
        
        try {
            this.startGeneration('Generating Digital Twin...');
            
            const formData = new FormData();
            
            // Add file with proper error handling
            const file = this.selfiesFileInput.files[0];
            if (!file) {
                this.handleGenerationFailed('No file selected');
                return;
            }
            
            formData.append('selfies_zip', file);
            formData.append('avatar_style', this.avatarStyle.value);
            formData.append('custom_prompt', this.customPrompt.value);
            formData.append('quality_mode', this.qualityModeGroup.querySelector('input:checked').value);
            
            // Show upload progress
            this.showUploadProgress('Uploading selfies...');
            
            const response = await fetch('/upload-selfies', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.sessionId = result.session_id;
                this.avatarJobId = result.job_id;
                this.showStatus('Upload successful! Starting avatar generation...', 'success');
                this.startStatusPolling();
            } else {
                const error = await response.json();
                this.handleGenerationFailed(error.error || 'Upload failed');
            }
            
        } catch (error) {
            this.handleGenerationFailed(`Upload failed: ${error.message}`);
        }
    }
    
    async generateFashionPhoto() {
        if (!this.validateForm()) return;
        
        try {
            this.startGeneration('Generating Fashion Photo...');
            
            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            formData.append('scene_prompt', this.scenePrompt.value);
            formData.append('quality_mode', this.fashionQualityModeGroup.querySelector('input:checked').value);
            
            // Add clothing items with proper error handling
            if (this.clothingItems.length === 0) {
                this.handleGenerationFailed('No clothing items selected');
                return;
            }
            
            this.clothingItems.forEach((item, index) => {
                formData.append(`clothing_files`, item.file);
                formData.append(`clothing_type_${index}`, item.type);
                formData.append(`clothing_layer_${index}`, item.layer);
                formData.append(`clothing_name_${index}`, item.name);
            });
            
            // Add pose control
            const poseControlType = this.poseControlType.value;
            if (poseControlType === 'reference' && this.referencePoseFileInput.files[0]) {
                formData.append('reference_pose_file', this.referencePoseFileInput.files[0]);
            } else if (poseControlType === 'preset' && this.posePreset.value) {
                formData.append('pose_preset', this.posePreset.value);
            }
            
            // Show upload progress
            this.showUploadProgress('Uploading clothing and scene...');
            
            const response = await fetch('/upload-clothing-scene', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.fashionJobId = result.job_id;
                this.showStatus('Upload successful! Starting fashion photo generation...', 'success');
                this.startStatusPolling();
            } else {
                const error = await response.json();
                this.handleGenerationFailed(error.error || 'Upload failed');
            }
            
        } catch (error) {
            this.handleGenerationFailed(`Upload failed: ${error.message}`);
        }
    }
    
    startGeneration(title) {
        this.progressTitle.textContent = title;
        this.progressContainer.style.display = 'block';
        this.smoothScrollTo(this.progressContainer);
    }
    
    startStatusPolling() {
        const jobId = this.fashionJobId || this.avatarJobId;
        if (!jobId) return;
        
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/status/${jobId}`);
                if (response.ok) {
                    const status = await response.json();
                    this.updateProgress(status);
                    
                    if (status.status === 'completed') {
                        clearInterval(pollInterval);
                        this.handleGenerationComplete(status);
                    } else if (status.status === 'failed') {
                        clearInterval(pollInterval);
                        this.handleGenerationFailed(status.message);
                    }
                }
            } catch (error) {
                console.error('Error polling status:', error);
            }
        }, 2000);
    }
    
    updateProgress(status) {
        this.progressFill.style.width = `${status.progress}%`;
        this.progressMessage.textContent = status.message;
    }
    
    handleGenerationComplete(status) {
        this.hideProgress();
        
        if (status.type === 'avatar') {
            this.advanceToStep(2);
            this.updateStepStatus(1, 'completed');
        } else if (status.type === 'fashion_photo') {
            this.advanceToStep(3);
            this.updateStepStatus(2, 'completed');
            this.displayResults(status.result);
        }
    }
    
    handleGenerationFailed(error) {
        this.hideProgress();
        this.showStatus(error, 'error');
    }
    
    advanceToStep(step) {
        this.currentStep = step;
        
        // Hide all steps
        this.step1.style.display = 'none';
        this.step2.style.display = 'none';
        this.step3.style.display = 'none';
        
        // Show current step
        if (step === 1) {
            this.step1.style.display = 'block';
        } else if (step === 2) {
            this.step2.style.display = 'block';
        } else if (step === 3) {
            this.step3.style.display = 'block';
        }
    }
    
    updateStepStatus(step, status) {
        const statusElement = document.getElementById(`step${step}-status`);
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `step-status ${status}`;
        }
    }
    
    displayResults(result) {
        this.resultsContainer.style.display = 'block';
        
        if (result.avatar_path) {
            this.avatarResult.src = result.avatar_path;
        }
        
        if (result.fashion_photo_path) {
            this.fashionResult.src = result.fashion_photo_path;
        }
    }
    
    async downloadFile(type) {
        try {
            const response = await fetch(`/download/${type}/${this.sessionId}`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${type}_result.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            }
        } catch (error) {
            this.showStatus('Download failed', 'error');
        }
    }
    
    regeneratePhoto() {
        this.advanceToStep(2);
        this.updateStepStatus(3, 'pending');
    }
    
    restartWorkflow() {
        this.currentStep = 1;
        this.sessionId = null;
        this.avatarJobId = null;
        this.fashionJobId = null;
        this.clothingItems = [];
        
        // Reset UI
        this.step1.style.display = 'block';
        this.step2.style.display = 'none';
        this.step3.style.display = 'none';
        this.resultsContainer.style.display = 'none';
        
        // Reset form elements
        this.selfiesFilePreview.style.display = 'none';
        this.clothingPreviewContainer.style.display = 'none';
        this.compatibilitySection.style.display = 'none';
        
        // Reset status
        this.updateStepStatus(1, 'pending');
        this.updateStepStatus(2, 'pending');
        this.updateStepStatus(3, 'pending');
        
        this.validateForm();
    }
    
    showProgress() {
        this.progressContainer.style.display = 'block';
    }
    
    hideProgress() {
        this.progressContainer.style.display = 'none';
    }
    
    showStatus(message, type = 'info') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message ${type}`;
        this.statusMessage.style.display = 'block';
        
        setTimeout(() => {
            this.statusMessage.style.display = 'none';
        }, 5000);
    }
    
    smoothScrollTo(element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showUploadProgress(message) {
        this.showStatus(message, 'info');
        // You can add a progress bar here if needed
    }
}

// Initialize the fashion workflow when the page loads
let fashionWorkflow;
document.addEventListener('DOMContentLoaded', () => {
    fashionWorkflow = new FashionWorkflow();
}); 