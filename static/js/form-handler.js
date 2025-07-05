/**
 * Enhanced Form Handler for Worldbuilding Features
 * 
 * Provides consistent form handling with:
 * - Real-time validation
 * - AI editing integration
 * - Progress tracking
 * - Error recovery
 * - Performance optimization
 */

class FormHandler {
    constructor() {
        this.activeRequests = new Map();
        this.editingMode = 'realtime'; // 'realtime' or 'fast'
        this.init();
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.bindFormEvents();
            this.setupEditingModeToggle();
        });
    }

    bindFormEvents() {
        // Handle all AI editing forms
        document.addEventListener('submit', (e) => {
            if (e.target.matches('.ai-edit-form')) {
                e.preventDefault();
                this.handleAIEditForm(e.target);
            }
        });

        // Handle tag-only saves
        document.addEventListener('click', (e) => {
            if (e.target.matches('.save-tags-btn')) {
                e.preventDefault();
                this.handleTagsSave(e.target);
            }
        });

        // Handle editing mode toggle
        document.addEventListener('change', (e) => {
            if (e.target.matches('.editing-mode-toggle')) {
                this.editingMode = e.target.value;
                this.updateEditingModeUI();
            }
        });
    }

    setupEditingModeToggle() {
        // Add editing mode toggle if it doesn't exist
        const editForms = document.querySelectorAll('.ai-edit-form');
        editForms.forEach(form => {
            if (!form.querySelector('.editing-mode-toggle')) {
                this.addEditingModeToggle(form);
            }
        });
    }

    addEditingModeToggle(form) {
        const toggleHTML = `
            <div class="mb-3">
                <label class="form-label">Editing Mode:</label>
                <div class="btn-group" role="group">
                    <input type="radio" class="btn-check editing-mode-toggle" name="editingMode" id="realtime-${Date.now()}" value="realtime" checked>
                    <label class="btn btn-outline-primary" for="realtime-${Date.now()}">
                        <i class="fas fa-eye me-1"></i>Real-time Preview
                    </label>
                    <input type="radio" class="btn-check editing-mode-toggle" name="editingMode" id="fast-${Date.now()}" value="fast">
                    <label class="btn btn-outline-success" for="fast-${Date.now()}">
                        <i class="fas fa-bolt me-1"></i>Fast Mode
                    </label>
                </div>
                <small class="form-text text-muted">
                    Real-time: See changes as they happen. Fast: Direct save without preview.
                </small>
            </div>
        `;

        const submitButton = form.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.insertAdjacentHTML('beforebegin', toggleHTML);
        }
    }

    updateEditingModeUI() {
        const statusElements = document.querySelectorAll('.editing-mode-status');
        statusElements.forEach(el => {
            el.textContent = this.editingMode === 'realtime' ? 'Real-time Preview Mode' : 'Fast Edit Mode';
            el.className = `editing-mode-status badge ${this.editingMode === 'realtime' ? 'bg-info' : 'bg-success'}`;
        });
    }

    async handleAIEditForm(form) {
        const formData = new FormData(form);
        const editRequest = formData.get('edit_request')?.trim();
        
        if (!editRequest) {
            window.apiUtils.showNotification('Please describe what you want to change.', 'warning');
            return;
        }

        const submitBtn = form.querySelector('button[type="submit"]');
        const entityType = this.getEntityTypeFromForm(form);
        const entityId = this.getEntityIdFromForm(form);

        if (!entityType || !entityId) {
            window.apiUtils.showNotification('Unable to identify entity for editing.', 'error');
            return;
        }

        // Prevent duplicate requests
        const requestKey = `${entityType}-${entityId}`;
        if (this.activeRequests.has(requestKey)) {
            window.apiUtils.showNotification('Edit already in progress...', 'info');
            return;
        }

        try {
            this.activeRequests.set(requestKey, true);
            window.apiUtils.setButtonLoading(submitBtn, true);

            if (this.editingMode === 'fast') {
                await this.performFastEdit(form, entityType, entityId, editRequest);
            } else {
                await this.performRealtimeEdit(form, entityType, entityId, editRequest);
            }

        } catch (error) {
            const message = window.apiUtils.handleError(error, 'AI editing');
            window.apiUtils.showNotification(message, 'danger');
        } finally {
            this.activeRequests.delete(requestKey);
            window.apiUtils.setButtonLoading(submitBtn, false);
        }
    }

    async performFastEdit(form, entityType, entityId, editRequest) {
        const novelId = this.getNovelIdFromForm(form);
        const url = `/novel/${novelId}/worldbuilding/${entityType}/${entityId}/edit/fast`;

        const response = await window.apiUtils.postJSON(url, {
            edit_request: editRequest
        });

        if (response.success) {
            this.updateEntityDisplay(response[entityType]);
            window.apiUtils.showNotification(response.message || 'Updated successfully!', 'success');
            form.querySelector('[name="edit_request"]').value = '';
        } else {
            throw new Error(response.error || 'Fast edit failed');
        }
    }

    async performRealtimeEdit(form, entityType, entityId, editRequest) {
        const novelId = this.getNovelIdFromForm(form);
        
        // Step 1: Get preview
        this.showEditingStatus('Generating preview...', 'info');
        const previewUrl = `/novel/${novelId}/worldbuilding/${entityType}/${entityId}/edit/preview`;
        
        const previewResponse = await window.apiUtils.postJSON(previewUrl, {
            edit_request: editRequest
        });

        if (previewResponse.error) {
            throw new Error(previewResponse.error);
        }

        // Step 2: Show real-time changes
        this.showRealTimeChanges(previewResponse);

        // Step 3: Auto-save after preview
        setTimeout(async () => {
            try {
                this.showEditingStatus('Saving changes...', 'warning');
                await this.saveEntityChanges(entityType, entityId, editRequest);
                this.showEditingStatus('Saved successfully!', 'success');
                form.querySelector('[name="edit_request"]').value = '';
            } catch (error) {
                this.showEditingStatus('Save failed', 'danger');
                throw error;
            }
        }, 3000);
    }

    async saveEntityChanges(entityType, entityId, editRequest) {
        const url = `/api/${entityType}/${entityId}/save`;
        const formData = `edit_request=${encodeURIComponent(editRequest)}`;

        const response = await window.apiUtils.postForm(url, formData);

        if (!response.success) {
            throw new Error(response.error || 'Save failed');
        }

        return response;
    }

    showRealTimeChanges(data) {
        // Highlight changed fields
        Object.keys(data).forEach(field => {
            const element = document.getElementById(field);
            if (element && data[field] !== undefined) {
                this.highlightChange(element, data[field]);
            }
        });
    }

    highlightChange(element, newValue) {
        // Update content
        if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
            element.value = newValue;
        } else {
            element.textContent = newValue;
        }

        // Add highlight effect
        element.style.color = '#0066cc';
        element.style.transition = 'color 0.3s ease';

        // Remove highlight after delay
        setTimeout(() => {
            element.style.color = '';
        }, 5000);
    }

    showEditingStatus(message, type) {
        let statusElement = document.querySelector('.editing-status');
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.className = 'editing-status';
            const form = document.querySelector('.ai-edit-form');
            if (form) {
                form.appendChild(statusElement);
            }
        }

        statusElement.innerHTML = `<span class="badge bg-${type}">${message}</span>`;
    }

    updateEntityDisplay(entityData) {
        // Update display fields with new data
        Object.keys(entityData).forEach(field => {
            const displayElement = document.getElementById(field);
            if (displayElement && entityData[field] !== undefined) {
                if (displayElement.tagName === 'INPUT' || displayElement.tagName === 'TEXTAREA') {
                    displayElement.value = entityData[field];
                } else {
                    displayElement.textContent = entityData[field];
                }
            }
        });
    }

    async handleTagsSave(button) {
        const form = button.closest('form');
        const entityType = this.getEntityTypeFromForm(form);
        const entityId = this.getEntityIdFromForm(form);
        const tagsInput = form.querySelector(`[name="${entityType}_tags"]`);

        if (!tagsInput) {
            window.apiUtils.showNotification('Tags input not found.', 'error');
            return;
        }

        try {
            window.apiUtils.setButtonLoading(button, true);

            const url = `/novel/${this.getNovelIdFromForm(form)}/worldbuilding/${entityType}/${entityId}/edit/tags`;
            const formData = `${entityType}_tags=${encodeURIComponent(tagsInput.value)}`;

            const response = await window.apiUtils.postForm(url, formData);

            if (response.success) {
                window.apiUtils.showNotification('Tags updated successfully!', 'success');
            } else {
                throw new Error(response.error || 'Failed to update tags');
            }

        } catch (error) {
            const message = window.apiUtils.handleError(error, 'updating tags');
            window.apiUtils.showNotification(message, 'danger');
        } finally {
            window.apiUtils.setButtonLoading(button, false);
        }
    }

    getEntityTypeFromForm(form) {
        // Extract entity type from form action or data attributes
        const action = form.action || window.location.pathname;
        if (action.includes('/characters/')) return 'characters';
        if (action.includes('/locations/')) return 'locations';
        if (action.includes('/lore/')) return 'lore';
        return null;
    }

    getEntityIdFromForm(form) {
        // Extract entity ID from form action or data attributes
        const action = form.action || window.location.pathname;
        const matches = action.match(/\/(characters|locations|lore)\/([^\/]+)/);
        return matches ? matches[2] : null;
    }

    getNovelIdFromForm(form) {
        // Extract novel ID from URL
        const pathParts = window.location.pathname.split('/');
        const novelIndex = pathParts.indexOf('novel');
        return novelIndex !== -1 && pathParts[novelIndex + 1] ? pathParts[novelIndex + 1] : null;
    }
}

// Initialize form handler
window.formHandler = new FormHandler();
