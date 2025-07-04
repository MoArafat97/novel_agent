/**
 * Undo Functionality JavaScript
 * 
 * Handles undo/redo operations for cross-reference changes with:
 * - Change history display
 * - Individual change undo
 * - Session-based undo
 * - Conflict detection and resolution
 */

class UndoManager {
    constructor() {
        this.changeHistory = [];
        this.currentNovelId = null;
        this.init();
    }

    init() {
        // Initialize when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            this.currentNovelId = this.getCurrentNovelId();
            this.bindUndoEvents();
            this.loadChangeHistory();
        });
    }

    bindUndoEvents() {
        // Bind undo button clicks
        document.addEventListener('click', (e) => {
            if (e.target.matches('.undo-change-btn')) {
                this.handleUndoChange(e);
            } else if (e.target.matches('.undo-session-btn')) {
                this.handleUndoSession(e);
            } else if (e.target.matches('.show-change-history-btn')) {
                this.showChangeHistoryModal();
            } else if (e.target.matches('.view-change-details-btn')) {
                this.handleViewChangeDetails(e);
            }
        });

        // Add undo button to cross-reference results if not present
        this.addUndoButtonToInterface();
    }

    addUndoButtonToInterface() {
        // Add undo button to entity detail pages
        const entityActions = document.querySelector('.btn-group');
        if (entityActions && !document.getElementById('showChangeHistoryBtn')) {
            const undoButton = document.createElement('button');
            undoButton.type = 'button';
            undoButton.className = 'btn btn-outline-warning show-change-history-btn';
            undoButton.id = 'showChangeHistoryBtn';
            undoButton.title = 'View Change History & Undo';
            undoButton.innerHTML = '<i class="fas fa-undo me-1"></i>History';
            
            entityActions.appendChild(undoButton);
        }
    }

    async loadChangeHistory() {
        if (!this.currentNovelId) return;

        try {
            const response = await fetch(`/novel/${this.currentNovelId}/change-history`);
            const data = await response.json();

            if (data.success) {
                this.changeHistory = data.history;
                this.updateUndoButtonState();
            }
        } catch (error) {
            console.error('Failed to load change history:', error);
        }
    }

    updateUndoButtonState() {
        const undoButton = document.getElementById('showChangeHistoryBtn');
        if (undoButton) {
            const hasHistory = this.changeHistory.length > 0;
            undoButton.disabled = !hasHistory;
            
            if (hasHistory) {
                undoButton.innerHTML = `<i class="fas fa-undo me-1"></i>History (${this.changeHistory.length})`;
            } else {
                undoButton.innerHTML = '<i class="fas fa-undo me-1"></i>History';
            }
        }
    }

    showChangeHistoryModal() {
        // Create or show change history modal
        let modal = document.getElementById('changeHistoryModal');
        if (!modal) {
            modal = this.createChangeHistoryModal();
            document.body.appendChild(modal);
        }

        // Populate with current history
        this.populateChangeHistoryModal();

        // Show modal
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();
    }

    createChangeHistoryModal() {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'changeHistoryModal';
        modal.tabIndex = -1;
        modal.innerHTML = `
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-history me-2"></i>
                            Change History & Undo
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Filter Controls -->
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="historyFilter" class="form-label">Filter by Type:</label>
                                    <select id="historyFilter" class="form-select">
                                        <option value="all">All Changes</option>
                                        <option value="cross_reference">Cross-Reference Updates</option>
                                        <option value="manual">Manual Edits</option>
                                        <option value="undo">Undo Operations</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="entityFilter" class="form-label">Filter by Entity:</label>
                                    <select id="entityFilter" class="form-select">
                                        <option value="all">All Entities</option>
                                        <option value="characters">Characters</option>
                                        <option value="locations">Locations</option>
                                        <option value="lore">Lore</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <!-- Change History List -->
                        <div id="changeHistoryList">
                            <!-- Content will be populated dynamically -->
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-warning" id="refreshHistoryBtn">
                            <i class="fas fa-refresh me-1"></i>Refresh
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Bind filter events
        modal.querySelector('#historyFilter').addEventListener('change', () => this.filterChangeHistory());
        modal.querySelector('#entityFilter').addEventListener('change', () => this.filterChangeHistory());
        modal.querySelector('#refreshHistoryBtn').addEventListener('click', () => this.refreshChangeHistory());

        return modal;
    }

    populateChangeHistoryModal() {
        const container = document.getElementById('changeHistoryList');
        if (!container) return;

        if (this.changeHistory.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="fas fa-history fa-3x text-muted mb-3"></i>
                    <h5>No Change History</h5>
                    <p class="text-muted">No changes have been made to this novel yet.</p>
                </div>
            `;
            return;
        }

        const html = this.changeHistory.map(session => this.buildSessionHTML(session)).join('');
        container.innerHTML = html;
    }

    buildSessionHTML(session) {
        const sessionDate = new Date(session.timestamp).toLocaleString();
        const changeCount = session.changes.length;
        
        return `
            <div class="card mb-3 change-session" data-session-id="${session.session_id}">
                <div class="card-header">
                    <div class="row align-items-center">
                        <div class="col">
                            <h6 class="mb-1">${session.description}</h6>
                            <small class="text-muted">
                                ${sessionDate} • ${changeCount} change${changeCount !== 1 ? 's' : ''}
                            </small>
                        </div>
                        <div class="col-auto">
                            <div class="btn-group btn-group-sm">
                                <button type="button" class="btn btn-outline-info view-session-details-btn" 
                                        data-session-id="${session.session_id}" title="View Details">
                                    <i class="fas fa-eye"></i>
                                </button>
                                <button type="button" class="btn btn-outline-warning undo-session-btn" 
                                        data-session-id="${session.session_id}" title="Undo Entire Session">
                                    <i class="fas fa-undo"></i> Undo All
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card-body collapse" id="session_${session.session_id}">
                    <div class="changes-list">
                        ${session.changes.map(change => this.buildChangeHTML(change)).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    buildChangeHTML(change) {
        const changeDate = new Date(change.timestamp).toLocaleString();
        const entityIcon = this.getEntityIcon(change.entity_type);
        
        return `
            <div class="change-item border-start border-3 border-primary ps-3 mb-3" data-change-id="${change.change_id}">
                <div class="row align-items-center">
                    <div class="col">
                        <div class="d-flex align-items-center mb-1">
                            <i class="${entityIcon} me-2"></i>
                            <strong>${change.description}</strong>
                        </div>
                        <div class="text-muted small">
                            ${change.entity_type} • ${changeDate}
                        </div>
                    </div>
                    <div class="col-auto">
                        <div class="btn-group btn-group-sm">
                            <button type="button" class="btn btn-outline-info view-change-details-btn" 
                                    data-change-id="${change.change_id}" title="View Change Details">
                                <i class="fas fa-eye"></i>
                            </button>
                            <button type="button" class="btn btn-outline-warning undo-change-btn" 
                                    data-change-id="${change.change_id}" title="Undo This Change">
                                <i class="fas fa-undo"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getEntityIcon(entityType) {
        const icons = {
            'characters': 'fas fa-user',
            'locations': 'fas fa-map-marker-alt',
            'lore': 'fas fa-book',
            'relationship': 'fas fa-link'
        };
        return icons[entityType] || 'fas fa-edit';
    }

    async handleUndoChange(event) {
        const changeId = event.target.closest('[data-change-id]').dataset.changeId;
        
        if (!confirm('Are you sure you want to undo this change? This action cannot be undone.')) {
            return;
        }

        try {
            const button = event.target.closest('.undo-change-btn');
            const originalText = button.innerHTML;
            
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            const response = await fetch(`/novel/${this.currentNovelId}/undo/change/${changeId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const result = await response.json();

            if (result.success) {
                this.showSuccess('Change undone successfully');
                await this.refreshChangeHistory();
                
                // Refresh the page to show updated content
                setTimeout(() => window.location.reload(), 1000);
            } else {
                this.showError(result.error || 'Failed to undo change');
                button.disabled = false;
                button.innerHTML = originalText;
            }

        } catch (error) {
            console.error('Undo change error:', error);
            this.showError('Failed to undo change');
        }
    }

    async handleUndoSession(event) {
        const sessionId = event.target.closest('[data-session-id]').dataset.sessionId;
        
        if (!confirm('Are you sure you want to undo this entire session? All changes in this session will be reverted. This action cannot be undone.')) {
            return;
        }

        try {
            const button = event.target.closest('.undo-session-btn');
            const originalText = button.innerHTML;
            
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            const response = await fetch(`/novel/${this.currentNovelId}/undo/session/${sessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const result = await response.json();

            if (result.success) {
                this.showSuccess('Session undone successfully');
                await this.refreshChangeHistory();
                
                // Refresh the page to show updated content
                setTimeout(() => window.location.reload(), 1000);
            } else {
                this.showError(result.error || 'Failed to undo session');
                button.disabled = false;
                button.innerHTML = originalText;
            }

        } catch (error) {
            console.error('Undo session error:', error);
            this.showError('Failed to undo session');
        }
    }

    handleViewChangeDetails(event) {
        const changeId = event.target.closest('[data-change-id]').dataset.changeId;
        const change = this.findChangeById(changeId);
        
        if (change) {
            this.showChangeDetailsModal(change);
        }
    }

    findChangeById(changeId) {
        for (const session of this.changeHistory) {
            for (const change of session.changes) {
                if (change.change_id === changeId) {
                    return change;
                }
            }
        }
        return null;
    }

    showChangeDetailsModal(change) {
        // Create or get details modal
        let modal = document.getElementById('changeDetailsModal');
        if (!modal) {
            modal = this.createChangeDetailsModal();
            document.body.appendChild(modal);
        }

        // Populate modal content
        this.populateChangeDetailsModal(change);

        // Show modal
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();
    }

    createChangeDetailsModal() {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'changeDetailsModal';
        modal.tabIndex = -1;
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-info-circle me-2"></i>
                            Change Details
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="changeDetailsBody">
                        <!-- Content will be populated dynamically -->
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-warning" id="undoFromDetailsBtn">
                            <i class="fas fa-undo me-1"></i>Undo This Change
                        </button>
                    </div>
                </div>
            </div>
        `;

        return modal;
    }

    populateChangeDetailsModal(change) {
        const modalBody = document.getElementById('changeDetailsBody');
        const undoBtn = document.getElementById('undoFromDetailsBtn');

        if (modalBody) {
            modalBody.innerHTML = `
                <div class="change-details">
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <strong>Change ID:</strong><br>
                            <code>${change.change_id}</code>
                        </div>
                        <div class="col-md-6">
                            <strong>Timestamp:</strong><br>
                            ${new Date(change.timestamp).toLocaleString()}
                        </div>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <strong>Operation:</strong><br>
                            ${change.operation_type}
                        </div>
                        <div class="col-md-6">
                            <strong>Entity:</strong><br>
                            ${change.entity_type} (${change.entity_id})
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <strong>Description:</strong><br>
                        ${change.description}
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Before State</h6>
                            <div class="card">
                                <div class="card-body">
                                    <pre class="mb-0" style="max-height: 300px; overflow-y: auto;">${JSON.stringify(change.before_state, null, 2)}</pre>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <h6>After State</h6>
                            <div class="card">
                                <div class="card-body">
                                    <pre class="mb-0" style="max-height: 300px; overflow-y: auto;">${JSON.stringify(change.after_state, null, 2)}</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Bind undo button
        if (undoBtn) {
            undoBtn.onclick = () => {
                bootstrap.Modal.getInstance(document.getElementById('changeDetailsModal')).hide();
                // Trigger undo for this change
                const fakeEvent = {
                    target: {
                        closest: () => ({ dataset: { changeId: change.change_id } })
                    }
                };
                this.handleUndoChange(fakeEvent);
            };
        }
    }

    async refreshChangeHistory() {
        await this.loadChangeHistory();
        this.populateChangeHistoryModal();
        this.updateUndoButtonState();
    }

    filterChangeHistory() {
        const typeFilter = document.getElementById('historyFilter')?.value || 'all';
        const entityFilter = document.getElementById('entityFilter')?.value || 'all';
        
        const sessions = document.querySelectorAll('.change-session');
        
        sessions.forEach(session => {
            const sessionId = session.dataset.sessionId;
            const sessionData = this.changeHistory.find(s => s.session_id === sessionId);
            
            if (!sessionData) {
                session.style.display = 'none';
                return;
            }
            
            let showSession = false;
            
            // Check if any change in the session matches the filters
            for (const change of sessionData.changes) {
                let matchesType = typeFilter === 'all' || 
                                change.operation_type.includes(typeFilter) ||
                                change.description.toLowerCase().includes(typeFilter);
                
                let matchesEntity = entityFilter === 'all' || 
                                  change.entity_type === entityFilter;
                
                if (matchesType && matchesEntity) {
                    showSession = true;
                    break;
                }
            }
            
            session.style.display = showSession ? 'block' : 'none';
        });
    }

    getCurrentNovelId() {
        // Extract novel ID from current URL
        const pathParts = window.location.pathname.split('/');
        const novelIndex = pathParts.indexOf('novel');
        return novelIndex !== -1 && pathParts[novelIndex + 1] ? pathParts[novelIndex + 1] : null;
    }

    showSuccess(message) {
        // Use existing success display logic or create a simple alert
        if (window.crossReferenceManager && window.crossReferenceManager.showSuccess) {
            window.crossReferenceManager.showSuccess(message);
        } else {
            alert(message);
        }
    }

    showError(message) {
        // Use existing error display logic or create a simple alert
        if (window.crossReferenceManager && window.crossReferenceManager.showError) {
            window.crossReferenceManager.showError(message);
        } else {
            alert(message);
        }
    }
}

// Initialize undo manager
const undoManager = new UndoManager();
