/**
 * Cross-Reference Analysis JavaScript
 * 
 * Handles cross-reference analysis workflow including:
 * - Triggering analysis
 * - Displaying results
 * - Managing user approval/rejection of updates
 * - Applying approved changes
 */

class CrossReferenceManager {
    constructor() {
        this.currentAnalysis = null;
        this.selectedUpdates = new Set();
        this.init();
    }

    init() {
        // Bind cross-reference button click events
        document.addEventListener('DOMContentLoaded', () => {
            const crossRefBtn = document.getElementById('crossReferenceBtn');
            if (crossRefBtn) {
                crossRefBtn.addEventListener('click', (e) => this.handleCrossReferenceClick(e));
            }
        });
    }

    async handleCrossReferenceClick(event) {
        const button = event.target.closest('#crossReferenceBtn');
        const entityType = button.dataset.entityType;
        const entityId = button.dataset.entityId;
        const novelId = button.dataset.novelId;

        if (!entityType || !entityId || !novelId) {
            this.showError('Missing required data for cross-reference analysis');
            return;
        }

        try {
            // Show loading state
            this.setButtonLoading(button, true);
            
            // Check agent status first
            const statusResponse = await fetch(`/novel/${novelId}/cross-reference/status`);
            const statusData = await statusResponse.json();
            
            if (!statusData.success || !statusData.available) {
                this.showError('Cross-reference analysis is not available. Please check your configuration.');
                return;
            }

            // Perform analysis
            const analysisResponse = await fetch(`/novel/${novelId}/cross-reference/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    entity_type: entityType,
                    entity_id: entityId
                })
            });

            const analysisData = await analysisResponse.json();

            if (analysisData.success) {
                this.currentAnalysis = analysisData.analysis;
                this.showAnalysisResults();
            } else {
                this.showError(analysisData.error || 'Analysis failed');
            }

        } catch (error) {
            console.error('Cross-reference analysis error:', error);
            this.showError('Failed to perform cross-reference analysis');
        } finally {
            this.setButtonLoading(button, false);
        }
    }

    setButtonLoading(button, loading) {
        if (loading) {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Analyzing...';
        } else {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-project-diagram me-1"></i>Cross-Reference';
        }
    }

    showAnalysisResults() {
        if (!this.currentAnalysis) return;

        // Check if we should use approval workflow
        const useApprovalWorkflow = this.shouldUseApprovalWorkflow();

        if (useApprovalWorkflow && window.approvalWorkflowManager) {
            this.showApprovalWorkflowResults();
        } else {
            this.showTraditionalResults();
        }
    }

    shouldUseApprovalWorkflow() {
        // Use approval workflow if there are suggested updates
        return this.currentAnalysis.suggested_updates &&
               this.currentAnalysis.suggested_updates.length > 0;
    }

    showApprovalWorkflowResults() {
        // Create and show modal with approval workflow
        const modal = this.createApprovalWorkflowModal();
        document.body.appendChild(modal);

        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();

        // Initialize approval workflow with updates
        if (window.approvalWorkflowManager) {
            const updates = this.prepareUpdatesForApproval();
            window.approvalWorkflowManager.displayUpdatesWithApproval(updates, 'approvalWorkflowContainer');
        }

        // Clean up modal when hidden
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
            if (window.approvalWorkflowManager) {
                window.approvalWorkflowManager.reset();
            }
        });
    }

    showTraditionalResults() {
        // Create and show traditional modal
        const modal = this.createResultsModal();
        document.body.appendChild(modal);

        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();

        // Clean up modal when hidden
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }

    createApprovalWorkflowModal() {
        const analysis = this.currentAnalysis;

        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'approvalWorkflowModal';
        modal.tabIndex = -1;
        modal.innerHTML = `
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-tasks me-2"></i>
                            Cross-Reference Analysis Results - ${analysis.entity_name}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Analysis Summary -->
                        <div class="alert alert-info mb-4">
                            <div class="row">
                                <div class="col-md-3 text-center">
                                    <div class="h5 mb-1">${analysis.verified_relationships?.length || 0}</div>
                                    <small>Relationships Found</small>
                                </div>
                                <div class="col-md-3 text-center">
                                    <div class="h5 mb-1">${analysis.suggested_updates?.length || 0}</div>
                                    <small>Suggested Updates</small>
                                </div>
                                <div class="col-md-3 text-center">
                                    <div class="h5 mb-1">${analysis.new_entities?.length || 0}</div>
                                    <small>New Entities</small>
                                </div>
                                <div class="col-md-3 text-center">
                                    <div class="h5 mb-1">${analysis.phase || 'N/A'}</div>
                                    <small>Analysis Phase</small>
                                </div>
                            </div>
                        </div>

                        <!-- Approval Workflow Container -->
                        <div id="approvalWorkflowContainer">
                            <!-- Content will be populated by approval workflow manager -->
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" id="finalApplyUpdatesBtn" disabled>
                            <i class="fas fa-save me-1"></i>Apply Selected Updates
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Bind final apply button
        const finalApplyBtn = modal.querySelector('#finalApplyUpdatesBtn');
        if (finalApplyBtn) {
            finalApplyBtn.addEventListener('click', () => this.applyApprovedUpdates());
        }

        return modal;
    }

    prepareUpdatesForApproval() {
        const analysis = this.currentAnalysis;
        const updates = [];

        // Add suggested updates
        if (analysis.suggested_updates) {
            analysis.suggested_updates.forEach((update, index) => {
                updates.push({
                    ...update,
                    type: 'Content Update',
                    confidence: update.confidence || 0.7,
                    description: update.description || 'Update entity content',
                    reasoning: update.reasoning || 'Based on cross-reference analysis',
                    evidence: update.evidence || 'Found in related entities',
                    entity_name: update.entity_name || analysis.entity_name,
                    entity_type: update.entity_type || analysis.entity_type,
                    update_id: `suggested_${index}`
                });
            });
        }

        // Add relationship updates
        if (analysis.verified_relationships) {
            analysis.verified_relationships.forEach((relationship, index) => {
                updates.push({
                    type: 'Relationship',
                    confidence: relationship.confidence || 0.8,
                    description: `Add relationship: ${relationship.relationship_type}`,
                    reasoning: relationship.reasoning || 'Verified relationship found',
                    evidence: relationship.evidence || 'Cross-reference analysis',
                    entity_name: relationship.target_entity_name || 'Related Entity',
                    entity_type: 'relationship',
                    relationship_type: relationship.relationship_type,
                    update_id: `relationship_${index}`,
                    current_state: {},
                    proposed_changes: relationship
                });
            });
        }

        // Add new entity suggestions
        if (analysis.new_entities) {
            analysis.new_entities.forEach((entity, index) => {
                updates.push({
                    type: 'New Entity',
                    confidence: entity.confidence || 0.6,
                    description: `Create new ${entity.entity_type}: ${entity.name}`,
                    reasoning: entity.reasoning || 'New entity detected in analysis',
                    evidence: entity.evidence || 'Found in content analysis',
                    entity_name: entity.name,
                    entity_type: entity.entity_type,
                    update_id: `new_entity_${index}`,
                    current_state: {},
                    proposed_changes: entity
                });
            });
        }

        return updates;
    }

    async applyApprovedUpdates() {
        if (!window.approvalWorkflowManager) {
            this.showError('Approval workflow not available');
            return;
        }

        const approvedUpdates = window.approvalWorkflowManager.getApprovedUpdates();

        if (approvedUpdates.length === 0) {
            this.showError('No updates selected for application');
            return;
        }

        try {
            // Show loading state
            const applyBtn = document.getElementById('finalApplyUpdatesBtn');
            if (applyBtn) {
                applyBtn.disabled = true;
                applyBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Applying Updates...';
            }

            // Apply updates using existing logic
            await this.applySelectedUpdates(approvedUpdates);

            // Show success message
            this.showSuccess(`Successfully applied ${approvedUpdates.length} update(s)`);

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('approvalWorkflowModal'));
            if (modal) {
                modal.hide();
            }

        } catch (error) {
            console.error('Error applying approved updates:', error);
            this.showError('Failed to apply updates');
        }
    }

    createResultsModal() {
        const analysis = this.currentAnalysis;
        const hasResults = (
            analysis.verified_relationships?.length > 0 ||
            analysis.suggested_updates?.length > 0 ||
            analysis.new_entities?.length > 0
        );

        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'crossReferenceModal';
        modal.tabIndex = -1;
        modal.innerHTML = `
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-project-diagram me-2"></i>
                            Cross-Reference Analysis Results
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        ${hasResults ? this.renderAnalysisContent() : this.renderNoResults()}
                    </div>
                    <div class="modal-footer">
                        ${hasResults ? this.renderModalFooter() : ''}
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;

        // Bind event handlers
        this.bindModalEvents(modal);
        
        return modal;
    }

    renderAnalysisContent() {
        const analysis = this.currentAnalysis;
        let content = '';

        // Entity being analyzed
        content += `
            <div class="alert alert-info">
                <h6><i class="fas fa-info-circle me-2"></i>Analyzing: ${analysis.entity_name}</h6>
                <small>Analysis completed at ${new Date(analysis.analysis_timestamp).toLocaleString()}</small>
            </div>
        `;

        // Verified relationships
        if (analysis.verified_relationships?.length > 0) {
            content += `
                <div class="mb-4">
                    <h6><i class="fas fa-link me-2"></i>Discovered Relationships (${analysis.verified_relationships.length})</h6>
                    <div class="row">
                        ${analysis.verified_relationships.map(rel => this.renderRelationship(rel)).join('')}
                    </div>
                </div>
            `;
        }

        // Suggested updates
        if (analysis.suggested_updates?.length > 0) {
            content += `
                <div class="mb-4">
                    <h6><i class="fas fa-edit me-2"></i>Suggested Updates (${analysis.suggested_updates.length})</h6>
                    <div class="accordion" id="updatesAccordion">
                        ${analysis.suggested_updates.map((update, index) => this.renderUpdate(update, index)).join('')}
                    </div>
                </div>
            `;
        }

        // New entities
        if (analysis.new_entities?.length > 0) {
            content += `
                <div class="mb-4">
                    <h6><i class="fas fa-plus-circle me-2"></i>Potential New Entities (${analysis.new_entities.length})</h6>
                    <div class="row">
                        ${analysis.new_entities.map(entity => this.renderNewEntity(entity)).join('')}
                    </div>
                </div>
            `;
        }

        return content;
    }

    renderNoResults() {
        return `
            <div class="text-center py-5">
                <i class="fas fa-search fa-3x text-muted mb-3"></i>
                <h5 class="text-muted">No Cross-References Found</h5>
                <p class="text-muted">
                    No significant relationships or connections were discovered for this entity.
                    This could mean the entity is well-isolated or the content doesn't contain
                    references to other entities in your novel.
                </p>
            </div>
        `;
    }

    renderRelationship(relationship) {
        // Handle both numeric and text confidence values
        let confidenceClass, confidenceText;

        if (typeof relationship.confidence === 'number') {
            // Numeric confidence (0.0 - 1.0)
            const conf = relationship.confidence;
            if (conf >= 0.8) {
                confidenceClass = 'success';
                confidenceText = `${(conf * 100).toFixed(0)}% (High)`;
            } else if (conf >= 0.6) {
                confidenceClass = 'warning';
                confidenceText = `${(conf * 100).toFixed(0)}% (Medium)`;
            } else {
                confidenceClass = 'secondary';
                confidenceText = `${(conf * 100).toFixed(0)}% (Low)`;
            }
        } else {
            // Text confidence (legacy)
            confidenceClass = {
                'high': 'success',
                'medium': 'warning',
                'low': 'secondary'
            }[relationship.confidence] || 'secondary';
            confidenceText = relationship.confidence;
        }

        return `
            <div class="col-md-6 mb-3">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">
                            ${relationship.target_entity_name}
                            <span class="badge bg-${confidenceClass} ms-2">${confidenceText}</span>
                        </h6>
                        <p class="card-text">
                            <strong>Relationship:</strong> ${relationship.relationship_type}<br>
                            <strong>Evidence:</strong> ${relationship.evidence}
                        </p>
                        ${relationship.new_information ? `
                            <div class="alert alert-light p-2 mt-2">
                                <small><strong>New Info:</strong> ${relationship.new_information}</small>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    renderUpdate(update, index) {
        const updateId = `update-${index}`;
        const confidenceClass = {
            'high': 'success',
            'medium': 'warning',
            'low': 'secondary'
        }[update.confidence] || 'secondary';

        return `
            <div class="accordion-item">
                <h2 class="accordion-header" id="heading-${index}">
                    <button class="accordion-button collapsed" type="button" 
                            data-bs-toggle="collapse" data-bs-target="#collapse-${index}">
                        <div class="form-check me-3">
                            <input class="form-check-input update-checkbox" type="checkbox" 
                                   id="check-${updateId}" data-update-index="${index}">
                        </div>
                        <div>
                            <strong>${update.target_entity_name}</strong>
                            <span class="badge bg-${confidenceClass} ms-2">${update.confidence}</span>
                            <br>
                            <small class="text-muted">${Object.keys(update.suggested_changes || {}).length} field(s) to update</small>
                        </div>
                    </button>
                </h2>
                <div id="collapse-${index}" class="accordion-collapse collapse" 
                     data-bs-parent="#updatesAccordion">
                    <div class="accordion-body">
                        ${this.renderUpdateDetails(update)}
                    </div>
                </div>
            </div>
        `;
    }

    renderUpdateDetails(update) {
        const changes = update.suggested_changes || {};
        let content = '';

        for (const [field, newValue] of Object.entries(changes)) {
            content += `
                <div class="mb-3">
                    <strong>${field.charAt(0).toUpperCase() + field.slice(1)}:</strong>
                    <div class="bg-light p-2 rounded mt-1">
                        ${Array.isArray(newValue) ? newValue.join(', ') : newValue}
                    </div>
                </div>
            `;
        }

        if (update.relationship) {
            content += `
                <div class="mt-3 pt-3 border-top">
                    <small class="text-muted">
                        <strong>Based on:</strong> ${update.relationship.relationship_type} - ${update.relationship.evidence}
                    </small>
                </div>
            `;
        }

        return content;
    }

    renderNewEntity(entity) {
        // Handle both numeric and text confidence values
        let confidenceClass, confidenceText;

        if (typeof entity.confidence === 'number') {
            // Numeric confidence (0.0 - 1.0)
            const conf = entity.confidence;
            if (conf >= 0.8) {
                confidenceClass = 'success';
                confidenceText = `${(conf * 100).toFixed(0)}%`;
            } else if (conf >= 0.6) {
                confidenceClass = 'warning';
                confidenceText = `${(conf * 100).toFixed(0)}%`;
            } else {
                confidenceClass = 'secondary';
                confidenceText = `${(conf * 100).toFixed(0)}%`;
            }
        } else {
            // Text confidence (legacy)
            confidenceClass = {
                'high': 'success',
                'medium': 'warning',
                'low': 'secondary'
            }[entity.confidence] || 'secondary';
            confidenceText = entity.confidence;
        }

        return `
            <div class="col-md-6 mb-3">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">
                            ${entity.name}
                            <span class="badge bg-primary ms-2">${entity.type}</span>
                            <span class="badge bg-${confidenceClass} ms-1">${confidenceText}</span>
                        </h6>
                        <p class="card-text">
                            ${entity.description}
                        </p>
                        <div class="alert alert-light p-2">
                            <small><strong>Evidence:</strong> "${entity.evidence}"</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderModalFooter() {
        return `
            <button type="button" class="btn btn-success" id="applySelectedBtn" disabled>
                <i class="fas fa-check me-1"></i>Apply Selected Updates
            </button>
        `;
    }

    bindModalEvents(modal) {
        // Handle update selection
        modal.addEventListener('change', (e) => {
            if (e.target.classList.contains('update-checkbox')) {
                this.handleUpdateSelection(e.target);
            }
        });

        // Handle apply button
        const applyBtn = modal.querySelector('#applySelectedBtn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.applySelectedUpdates());
        }
    }

    handleUpdateSelection(checkbox) {
        const updateIndex = parseInt(checkbox.dataset.updateIndex);
        
        if (checkbox.checked) {
            this.selectedUpdates.add(updateIndex);
        } else {
            this.selectedUpdates.delete(updateIndex);
        }

        // Update apply button state
        const applyBtn = document.getElementById('applySelectedBtn');
        if (applyBtn) {
            applyBtn.disabled = this.selectedUpdates.size === 0;
            applyBtn.innerHTML = `
                <i class="fas fa-check me-1"></i>
                Apply Selected Updates (${this.selectedUpdates.size})
            `;
        }
    }

    async applySelectedUpdates() {
        if (this.selectedUpdates.size === 0) return;

        const applyBtn = document.getElementById('applySelectedBtn');
        const originalText = applyBtn.innerHTML;

        try {
            // Show loading state
            applyBtn.disabled = true;
            applyBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Applying Updates...';

            // Prepare updates to apply
            const updatesToApply = Array.from(this.selectedUpdates).map(index => 
                this.currentAnalysis.suggested_updates[index]
            );

            // Apply updates
            const response = await fetch(`/novel/${this.currentAnalysis.novel_id}/cross-reference/apply`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    updates: updatesToApply
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showSuccess(`Successfully applied ${result.total_applied} updates!`);
                
                // Close modal and refresh page
                const modal = bootstrap.Modal.getInstance(document.getElementById('crossReferenceModal'));
                modal.hide();
                
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
            } else {
                this.showError(result.error || 'Failed to apply updates');
            }

        } catch (error) {
            console.error('Apply updates error:', error);
            this.showError('Failed to apply updates');
        } finally {
            applyBtn.disabled = false;
            applyBtn.innerHTML = originalText;
        }
    }

    showError(message) {
        this.showAlert(message, 'danger');
    }

    showSuccess(message) {
        this.showAlert(message, 'success');
    }

    showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }
    // Enhanced confidence visualization utilities
    getConfidenceBadge(confidence, showIcon = true) {
        const percentage = Math.round(confidence * 100);
        let badgeClass = 'bg-secondary';
        let icon = 'fas fa-question';
        let label = 'Unknown';

        if (confidence >= 0.8) {
            badgeClass = 'bg-success';
            icon = 'fas fa-check-circle';
            label = 'High';
        } else if (confidence >= 0.6) {
            badgeClass = 'bg-warning text-dark';
            icon = 'fas fa-exclamation-triangle';
            label = 'Medium';
        } else if (confidence >= 0.4) {
            badgeClass = 'bg-danger';
            icon = 'fas fa-times-circle';
            label = 'Low';
        } else {
            badgeClass = 'bg-secondary';
            icon = 'fas fa-question-circle';
            label = 'Very Low';
        }

        const iconHtml = showIcon ? `<i class="${icon} me-1"></i>` : '';
        return `<span class="badge ${badgeClass}" title="Confidence: ${percentage}%">
                    ${iconHtml}${percentage}% (${label})
                </span>`;
    }

    getConfidenceProgressBar(confidence, height = '8px') {
        const percentage = Math.round(confidence * 100);
        let progressClass = 'bg-secondary';

        if (confidence >= 0.8) {
            progressClass = 'bg-success';
        } else if (confidence >= 0.6) {
            progressClass = 'bg-warning';
        } else if (confidence >= 0.4) {
            progressClass = 'bg-danger';
        }

        return `
            <div class="progress mb-2" style="height: ${height};">
                <div class="progress-bar ${progressClass}" role="progressbar"
                     style="width: ${percentage}%" aria-valuenow="${percentage}"
                     aria-valuemin="0" aria-valuemax="100" title="Confidence: ${percentage}%">
                </div>
            </div>
        `;
    }

    addConfidenceVisualization() {
        // Add confidence indicators to existing results
        document.querySelectorAll('.badge').forEach(badge => {
            if (badge.textContent.includes('%')) {
                // Add visual enhancement to existing confidence badges
                badge.style.fontSize = '0.8em';
                badge.style.fontWeight = 'bold';

                // Add hover effect
                badge.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.1)';
                    this.style.transition = 'transform 0.2s';
                });

                badge.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1)';
                });
            }
        });
    }

    highlightEvidence(text, entityName, evidenceTerms = []) {
        if (!text || !entityName) return text;

        let highlightedText = text;

        // Highlight the main entity name
        const entityRegex = new RegExp(`\\b${this.escapeRegex(entityName)}\\b`, 'gi');
        highlightedText = highlightedText.replace(entityRegex,
            `<mark class="entity-highlight" style="background-color: #fff3cd; padding: 1px 3px; border-radius: 3px;">$&</mark>`);

        // Highlight evidence terms
        evidenceTerms.forEach(term => {
            if (term && term.length > 2) {
                const termRegex = new RegExp(`\\b${this.escapeRegex(term)}\\b`, 'gi');
                highlightedText = highlightedText.replace(termRegex,
                    `<mark class="evidence-highlight" style="background-color: #d1ecf1; padding: 1px 3px; border-radius: 3px;">$&</mark>`);
            }
        });

        return highlightedText;
    }

    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    addEvidenceHighlighting() {
        // Add evidence highlighting to relationship descriptions
        document.querySelectorAll('.relationship-evidence, .entity-context').forEach(element => {
            const text = element.textContent;
            const entityName = element.dataset.entityName || '';
            const evidence = element.dataset.evidence ? element.dataset.evidence.split(',') : [];

            if (text && entityName) {
                element.innerHTML = this.highlightEvidence(text, entityName, evidence);
            }
        });
    }

    createComparisonView(beforeData, afterData, title = "Comparison") {
        return `
            <div class="comparison-view">
                <h6>${title}</h6>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-light">
                                <small class="text-muted">
                                    <i class="fas fa-arrow-left me-1"></i>Before
                                </small>
                            </div>
                            <div class="card-body">
                                <pre class="mb-0" style="font-size: 0.85em; max-height: 200px; overflow-y: auto;">${this.formatComparisonData(beforeData)}</pre>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-light">
                                <small class="text-muted">
                                    <i class="fas fa-arrow-right me-1"></i>After
                                </small>
                            </div>
                            <div class="card-body">
                                <pre class="mb-0" style="font-size: 0.85em; max-height: 200px; overflow-y: auto;">${this.formatComparisonData(afterData)}</pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    formatComparisonData(data) {
        if (typeof data === 'string') {
            return data;
        } else if (typeof data === 'object') {
            return JSON.stringify(data, null, 2);
        }
        return String(data);
    }

    addComparisonViews() {
        // Add comparison views to update previews
        document.querySelectorAll('.update-preview').forEach(element => {
            const beforeData = element.dataset.beforeState;
            const afterData = element.dataset.afterState;

            if (beforeData && afterData) {
                try {
                    const before = JSON.parse(beforeData);
                    const after = JSON.parse(afterData);
                    const comparisonHtml = this.createComparisonView(before, after, "Update Preview");

                    element.innerHTML = comparisonHtml;
                } catch (error) {
                    console.warn('Failed to parse comparison data:', error);
                }
            }
        });
    }
}

// Initialize cross-reference manager
const crossReferenceManager = new CrossReferenceManager();
