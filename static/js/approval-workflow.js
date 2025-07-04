/**
 * Approval Workflow JavaScript
 * 
 * Handles interactive approval/rejection of cross-reference updates with:
 * - Individual update approval/rejection
 * - Bulk operations
 * - Preview changes
 * - Confidence-based recommendations
 */

class ApprovalWorkflowManager {
    constructor() {
        this.pendingUpdates = new Map();
        this.approvedUpdates = new Set();
        this.rejectedUpdates = new Set();
        this.previewData = new Map();
        this.init();
    }

    init() {
        // Initialize when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            this.bindApprovalEvents();
        });
    }

    bindApprovalEvents() {
        // Bind approval/rejection buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('.approve-update-btn')) {
                this.handleApproveUpdate(e);
            } else if (e.target.matches('.reject-update-btn')) {
                this.handleRejectUpdate(e);
            } else if (e.target.matches('.preview-update-btn')) {
                this.handlePreviewUpdate(e);
            } else if (e.target.matches('.bulk-approve-btn')) {
                this.handleBulkApprove(e);
            } else if (e.target.matches('.bulk-reject-btn')) {
                this.handleBulkReject(e);
            }
        });

        // Bind confidence filter
        const confidenceFilter = document.getElementById('confidenceFilter');
        if (confidenceFilter) {
            confidenceFilter.addEventListener('change', (e) => this.handleConfidenceFilter(e));
        }

        // Bind select all checkbox
        const selectAllCheckbox = document.getElementById('selectAllUpdates');
        if (selectAllCheckbox) {
            selectAllCheckbox.addEventListener('change', (e) => this.handleSelectAll(e));
        }
    }

    displayUpdatesWithApproval(updates, containerId = 'crossReferenceResults') {
        const container = document.getElementById(containerId);
        if (!container || !updates || updates.length === 0) {
            return;
        }

        // Store updates for processing
        updates.forEach((update, index) => {
            this.pendingUpdates.set(index, update);
        });

        // Build approval interface HTML
        const html = this.buildApprovalInterfaceHTML(updates);
        container.innerHTML = html;

        // Initialize approval state
        this.updateApprovalSummary();
    }

    buildApprovalInterfaceHTML(updates) {
        const html = `
            <div class="approval-workflow">
                <!-- Approval Controls -->
                <div class="card mb-4">
                    <div class="card-header">
                        <div class="row align-items-center">
                            <div class="col">
                                <h5 class="mb-0">
                                    <i class="fas fa-tasks me-2"></i>
                                    Cross-Reference Updates Approval
                                </h5>
                            </div>
                            <div class="col-auto">
                                <div class="btn-group" role="group">
                                    <button type="button" class="btn btn-success bulk-approve-btn">
                                        <i class="fas fa-check-double me-1"></i>Approve All
                                    </button>
                                    <button type="button" class="btn btn-danger bulk-reject-btn">
                                        <i class="fas fa-times-circle me-1"></i>Reject All
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="confidenceFilter" class="form-label">Filter by Confidence:</label>
                                    <select id="confidenceFilter" class="form-select">
                                        <option value="all">All Updates</option>
                                        <option value="high">High Confidence (≥80%)</option>
                                        <option value="medium">Medium Confidence (60-79%)</option>
                                        <option value="low">Low Confidence (<60%)</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="form-check mt-4">
                                    <input class="form-check-input" type="checkbox" id="selectAllUpdates">
                                    <label class="form-check-label" for="selectAllUpdates">
                                        Select All Visible Updates
                                    </label>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Approval Summary -->
                        <div id="approvalSummary" class="mt-3">
                            <div class="row text-center">
                                <div class="col-md-4">
                                    <div class="text-muted">Pending</div>
                                    <div class="h4 text-warning" id="pendingCount">${updates.length}</div>
                                </div>
                                <div class="col-md-4">
                                    <div class="text-muted">Approved</div>
                                    <div class="h4 text-success" id="approvedCount">0</div>
                                </div>
                                <div class="col-md-4">
                                    <div class="text-muted">Rejected</div>
                                    <div class="h4 text-danger" id="rejectedCount">0</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Updates List -->
                <div id="updatesList">
                    ${this.buildUpdatesListHTML(updates)}
                </div>

                <!-- Apply Changes Button -->
                <div class="text-center mt-4">
                    <button id="applyApprovedUpdatesBtn" type="button" class="btn btn-primary btn-lg" disabled>
                        <i class="fas fa-save me-2"></i>Apply Approved Updates
                    </button>
                </div>
            </div>
        `;

        return html;
    }

    buildUpdatesListHTML(updates) {
        return updates.map((update, index) => {
            const confidence = update.confidence || 0;
            const confidenceClass = this.getConfidenceClass(confidence);
            const confidenceText = this.getConfidenceText(confidence);

            return `
                <div class="card mb-3 update-item" data-update-index="${index}" data-confidence="${confidence}">
                    <div class="card-header">
                        <div class="row align-items-center">
                            <div class="col">
                                <div class="form-check">
                                    <input class="form-check-input update-checkbox" type="checkbox" 
                                           id="update_${index}" data-update-index="${index}">
                                    <label class="form-check-label" for="update_${index}">
                                        <strong>${update.type || 'Update'}</strong>
                                        ${update.entity_name ? `- ${update.entity_name}` : ''}
                                    </label>
                                </div>
                            </div>
                            <div class="col-auto">
                                <span class="badge ${confidenceClass} me-2">${confidenceText}</span>
                                <div class="btn-group btn-group-sm" role="group">
                                    <button type="button" class="btn btn-outline-info preview-update-btn" 
                                            data-update-index="${index}" title="Preview Changes">
                                        <i class="fas fa-eye"></i>
                                    </button>
                                    <button type="button" class="btn btn-outline-success approve-update-btn" 
                                            data-update-index="${index}" title="Approve Update">
                                        <i class="fas fa-check"></i>
                                    </button>
                                    <button type="button" class="btn btn-outline-danger reject-update-btn" 
                                            data-update-index="${index}" title="Reject Update">
                                        <i class="fas fa-times"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-8">
                                <p class="mb-2"><strong>Description:</strong> ${update.description || 'No description available'}</p>
                                ${update.reasoning ? `<p class="mb-2"><strong>Reasoning:</strong> ${update.reasoning}</p>` : ''}
                                ${update.evidence ? `<p class="mb-0"><strong>Evidence:</strong> ${update.evidence}</p>` : ''}
                            </div>
                            <div class="col-md-4">
                                <div class="approval-status" id="status_${index}">
                                    <span class="badge bg-warning">Pending Review</span>
                                </div>
                                ${this.buildUpdateMetadataHTML(update)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    buildUpdateMetadataHTML(update) {
        const metadata = [];
        
        if (update.entity_type) {
            metadata.push(`<small class="text-muted">Type: ${update.entity_type}</small>`);
        }
        
        if (update.relationship_type) {
            metadata.push(`<small class="text-muted">Relationship: ${update.relationship_type}</small>`);
        }
        
        if (update.impact_score) {
            metadata.push(`<small class="text-muted">Impact: ${update.impact_score}/10</small>`);
        }

        return metadata.length > 0 ? `<div class="mt-2">${metadata.join('<br>')}</div>` : '';
    }

    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'bg-success';
        if (confidence >= 0.6) return 'bg-warning';
        return 'bg-danger';
    }

    getConfidenceText(confidence) {
        return `${Math.round(confidence * 100)}% Confidence`;
    }

    handleApproveUpdate(event) {
        const updateIndex = parseInt(event.target.closest('[data-update-index]').dataset.updateIndex);
        this.approveUpdate(updateIndex);
    }

    handleRejectUpdate(event) {
        const updateIndex = parseInt(event.target.closest('[data-update-index]').dataset.updateIndex);
        this.rejectUpdate(updateIndex);
    }

    handlePreviewUpdate(event) {
        const updateIndex = parseInt(event.target.closest('[data-update-index]').dataset.updateIndex);
        this.previewUpdate(updateIndex);
    }

    handleBulkApprove(event) {
        const visibleUpdates = this.getVisibleUpdates();
        visibleUpdates.forEach(index => this.approveUpdate(index));
    }

    handleBulkReject(event) {
        const visibleUpdates = this.getVisibleUpdates();
        visibleUpdates.forEach(index => this.rejectUpdate(index));
    }

    handleConfidenceFilter(event) {
        const filterValue = event.target.value;
        this.filterUpdatesByConfidence(filterValue);
    }

    handleSelectAll(event) {
        const isChecked = event.target.checked;
        const visibleCheckboxes = document.querySelectorAll('.update-item:not([style*="display: none"]) .update-checkbox');
        
        visibleCheckboxes.forEach(checkbox => {
            checkbox.checked = isChecked;
        });
    }

    approveUpdate(updateIndex) {
        this.approvedUpdates.add(updateIndex);
        this.rejectedUpdates.delete(updateIndex);
        this.updateUpdateStatus(updateIndex, 'approved');
        this.updateApprovalSummary();
    }

    rejectUpdate(updateIndex) {
        this.rejectedUpdates.add(updateIndex);
        this.approvedUpdates.delete(updateIndex);
        this.updateUpdateStatus(updateIndex, 'rejected');
        this.updateApprovalSummary();
    }

    updateUpdateStatus(updateIndex, status) {
        const statusElement = document.getElementById(`status_${updateIndex}`);
        if (statusElement) {
            let badgeClass, statusText;
            
            switch (status) {
                case 'approved':
                    badgeClass = 'bg-success';
                    statusText = 'Approved';
                    break;
                case 'rejected':
                    badgeClass = 'bg-danger';
                    statusText = 'Rejected';
                    break;
                default:
                    badgeClass = 'bg-warning';
                    statusText = 'Pending Review';
            }
            
            statusElement.innerHTML = `<span class="badge ${badgeClass}">${statusText}</span>`;
        }

        // Update card styling
        const updateCard = document.querySelector(`[data-update-index="${updateIndex}"]`);
        if (updateCard) {
            updateCard.classList.remove('border-success', 'border-danger');
            if (status === 'approved') {
                updateCard.classList.add('border-success');
            } else if (status === 'rejected') {
                updateCard.classList.add('border-danger');
            }
        }
    }

    updateApprovalSummary() {
        const totalUpdates = this.pendingUpdates.size;
        const approvedCount = this.approvedUpdates.size;
        const rejectedCount = this.rejectedUpdates.size;
        const pendingCount = totalUpdates - approvedCount - rejectedCount;

        // Update counters
        const pendingEl = document.getElementById('pendingCount');
        const approvedEl = document.getElementById('approvedCount');
        const rejectedEl = document.getElementById('rejectedCount');

        if (pendingEl) pendingEl.textContent = pendingCount;
        if (approvedEl) approvedEl.textContent = approvedCount;
        if (rejectedEl) rejectedEl.textContent = rejectedCount;

        // Enable/disable apply button
        const applyBtn = document.getElementById('applyApprovedUpdatesBtn');
        if (applyBtn) {
            applyBtn.disabled = approvedCount === 0;
            applyBtn.innerHTML = `<i class="fas fa-save me-2"></i>Apply ${approvedCount} Approved Update${approvedCount !== 1 ? 's' : ''}`;
        }
    }

    filterUpdatesByConfidence(filterValue) {
        const updateItems = document.querySelectorAll('.update-item');
        
        updateItems.forEach(item => {
            const confidence = parseFloat(item.dataset.confidence);
            let show = true;

            switch (filterValue) {
                case 'high':
                    show = confidence >= 0.8;
                    break;
                case 'medium':
                    show = confidence >= 0.6 && confidence < 0.8;
                    break;
                case 'low':
                    show = confidence < 0.6;
                    break;
                case 'all':
                default:
                    show = true;
            }

            item.style.display = show ? 'block' : 'none';
        });

        // Update select all checkbox
        const selectAllCheckbox = document.getElementById('selectAllUpdates');
        if (selectAllCheckbox) {
            selectAllCheckbox.checked = false;
        }
    }

    getVisibleUpdates() {
        const visibleItems = document.querySelectorAll('.update-item:not([style*="display: none"])');
        return Array.from(visibleItems).map(item => parseInt(item.dataset.updateIndex));
    }

    previewUpdate(updateIndex) {
        const update = this.pendingUpdates.get(updateIndex);
        if (!update) return;

        // Create preview modal
        this.showPreviewModal(update, updateIndex);
    }

    showPreviewModal(update, updateIndex) {
        // Create or get preview modal
        let modal = document.getElementById('updatePreviewModal');
        if (!modal) {
            modal = this.createPreviewModal();
            document.body.appendChild(modal);
        }

        // Populate modal content
        this.populatePreviewModal(update, updateIndex);

        // Show modal
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();
    }

    createPreviewModal() {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'updatePreviewModal';
        modal.tabIndex = -1;
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-eye me-2"></i>
                            Preview Update Changes
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="previewModalBody">
                        <!-- Content will be populated dynamically -->
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-danger" id="previewRejectBtn">
                            <i class="fas fa-times me-1"></i>Reject
                        </button>
                        <button type="button" class="btn btn-success" id="previewApproveBtn">
                            <i class="fas fa-check me-1"></i>Approve
                        </button>
                    </div>
                </div>
            </div>
        `;

        return modal;
    }

    populatePreviewModal(update, updateIndex) {
        const modalBody = document.getElementById('previewModalBody');
        const approveBtn = document.getElementById('previewApproveBtn');
        const rejectBtn = document.getElementById('previewRejectBtn');

        if (modalBody) {
            modalBody.innerHTML = `
                <div class="preview-content">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Current State</h6>
                            <div class="card">
                                <div class="card-body">
                                    <pre class="mb-0">${this.formatCurrentState(update)}</pre>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h6>Proposed Changes</h6>
                            <div class="card">
                                <div class="card-body">
                                    <pre class="mb-0">${this.formatProposedChanges(update)}</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mt-3">
                        <h6>Update Details</h6>
                        <table class="table table-sm">
                            <tr><td><strong>Type:</strong></td><td>${update.type || 'N/A'}</td></tr>
                            <tr><td><strong>Confidence:</strong></td><td>${Math.round((update.confidence || 0) * 100)}%</td></tr>
                            <tr><td><strong>Description:</strong></td><td>${update.description || 'N/A'}</td></tr>
                            <tr><td><strong>Reasoning:</strong></td><td>${update.reasoning || 'N/A'}</td></tr>
                        </table>
                    </div>
                </div>
            `;
        }

        // Bind approve/reject buttons
        if (approveBtn) {
            approveBtn.onclick = () => {
                this.approveUpdate(updateIndex);
                bootstrap.Modal.getInstance(document.getElementById('updatePreviewModal')).hide();
            };
        }

        if (rejectBtn) {
            rejectBtn.onclick = () => {
                this.rejectUpdate(updateIndex);
                bootstrap.Modal.getInstance(document.getElementById('updatePreviewModal')).hide();
            };
        }
    }

    formatCurrentState(update) {
        // Format current state for display
        return JSON.stringify(update.current_state || {}, null, 2);
    }

    formatProposedChanges(update) {
        // Format proposed changes for display
        return JSON.stringify(update.proposed_changes || update, null, 2);
    }

    getApprovedUpdates() {
        const approved = [];
        this.approvedUpdates.forEach(index => {
            const update = this.pendingUpdates.get(index);
            if (update) {
                approved.push({ ...update, index });
            }
        });
        return approved;
    }

    getRejectedUpdates() {
        const rejected = [];
        this.rejectedUpdates.forEach(index => {
            const update = this.pendingUpdates.get(index);
            if (update) {
                rejected.push({ ...update, index });
            }
        });
        return rejected;
    }

    reset() {
        this.pendingUpdates.clear();
        this.approvedUpdates.clear();
        this.rejectedUpdates.clear();
        this.previewData.clear();
    }
}

// Initialize approval workflow manager
const approvalWorkflowManager = new ApprovalWorkflowManager();
