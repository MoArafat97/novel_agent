/**
 * Streaming Cross-Reference Analysis JavaScript
 * 
 * Handles real-time streaming cross-reference analysis with:
 * - Progress indicators
 * - Live status updates
 * - Cancellation support
 * - Real-time results display
 */

class StreamingCrossReferenceManager {
    constructor() {
        this.currentJobId = null;
        this.eventSource = null;
        this.isAnalyzing = false;
        this.selectedUpdates = new Set();
        this.init();
    }

    init() {
        // Bind streaming cross-reference button click events
        document.addEventListener('DOMContentLoaded', () => {
            const streamingBtn = document.getElementById('streamingCrossReferenceBtn');
            if (streamingBtn) {
                streamingBtn.addEventListener('click', (e) => this.handleStreamingAnalysisClick(e));
            }
        });
    }

    async handleStreamingAnalysisClick(event) {
        const button = event.target.closest('#streamingCrossReferenceBtn');
        const entityType = button.dataset.entityType;
        const entityId = button.dataset.entityId;
        const novelId = button.dataset.novelId;

        if (!entityType || !entityId || !novelId) {
            this.showError('Missing required data for cross-reference analysis');
            return;
        }

        try {
            // Show streaming modal
            this.showStreamingModal();
            
            // Check agent status first
            const statusResponse = await fetch(`/novel/${novelId}/cross-reference/status`);
            const statusData = await statusResponse.json();
            
            if (!statusData.success || !statusData.available) {
                this.showError('Cross-reference analysis is not available. Please check your configuration.');
                return;
            }

            // Start streaming analysis
            const analysisResponse = await fetch(`/novel/${novelId}/cross-reference/analyze-stream`, {
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
                this.currentJobId = analysisData.job_id;
                this.startProgressStream(analysisData.stream_url);
            } else {
                this.showError(analysisData.error || 'Failed to start analysis');
            }

        } catch (error) {
            console.error('Streaming cross-reference analysis error:', error);
            const message = window.apiUtils ?
                window.apiUtils.handleError(error, 'streaming analysis') :
                'Failed to start streaming analysis';
            this.showError(message);
        }
    }

    showStreamingModal() {
        // Create or show streaming modal
        let modal = document.getElementById('streamingCrossReferenceModal');
        if (!modal) {
            modal = this.createStreamingModal();
            document.body.appendChild(modal);
        }

        // Reset modal state
        this.resetModalState();
        
        // Show modal
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();
        
        this.isAnalyzing = true;
    }

    createStreamingModal() {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'streamingCrossReferenceModal';
        modal.tabIndex = -1;
        modal.innerHTML = `
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-project-diagram me-2"></i>
                            Cross-Reference Analysis
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Progress Section -->
                        <div id="progressSection">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h6 class="mb-0">Analysis Progress</h6>
                                <button id="cancelAnalysisBtn" class="btn btn-outline-danger btn-sm">
                                    <i class="fas fa-times me-1"></i>Cancel
                                </button>
                            </div>
                            
                            <!-- Progress Bar -->
                            <div class="progress mb-3" style="height: 25px;">
                                <div id="analysisProgressBar" class="progress-bar progress-bar-striped progress-bar-animated" 
                                     role="progressbar" style="width: 0%">
                                    <span id="progressText">Initializing...</span>
                                </div>
                            </div>
                            
                            <!-- Current Stage -->
                            <div class="card">
                                <div class="card-body">
                                    <div class="d-flex align-items-center">
                                        <div id="stageIcon" class="me-3">
                                            <i class="fas fa-spinner fa-spin text-primary"></i>
                                        </div>
                                        <div>
                                            <h6 id="currentStage" class="mb-1">Initializing Analysis</h6>
                                            <p id="stageDescription" class="mb-0 text-muted">
                                                Preparing to analyze entity relationships...
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Live Updates -->
                            <div id="liveUpdates" class="mt-3" style="display: none;">
                                <h6>Live Updates</h6>
                                <div id="updatesList" class="list-group" style="max-height: 200px; overflow-y: auto;">
                                </div>
                            </div>
                        </div>

                        <!-- Results Section (hidden initially) -->
                        <div id="resultsSection" style="display: none;">
                            <div id="analysisResults"></div>
                        </div>

                        <!-- Error Section (hidden initially) -->
                        <div id="errorSection" style="display: none;">
                            <div class="alert alert-danger">
                                <h6><i class="fas fa-exclamation-triangle me-2"></i>Analysis Failed</h6>
                                <p id="errorMessage" class="mb-0"></p>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <div id="progressFooter">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                        <div id="resultsFooter" style="display: none;">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button id="applySelectedUpdatesBtn" type="button" class="btn btn-primary">
                                <i class="fas fa-check me-1"></i>Apply Selected Updates
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Bind cancel button
        modal.querySelector('#cancelAnalysisBtn').addEventListener('click', () => this.cancelAnalysis());
        
        // Bind apply updates button
        modal.querySelector('#applySelectedUpdatesBtn').addEventListener('click', () => this.applySelectedUpdates());

        return modal;
    }

    resetModalState() {
        // Reset progress
        const progressBar = document.getElementById('analysisProgressBar');
        const progressText = document.getElementById('progressText');
        const currentStage = document.getElementById('currentStage');
        const stageDescription = document.getElementById('stageDescription');
        const stageIcon = document.getElementById('stageIcon');

        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated';
        }
        if (progressText) progressText.textContent = 'Initializing...';
        if (currentStage) currentStage.textContent = 'Initializing Analysis';
        if (stageDescription) stageDescription.textContent = 'Preparing to analyze entity relationships...';
        if (stageIcon) stageIcon.innerHTML = '<i class="fas fa-spinner fa-spin text-primary"></i>';

        // Show/hide sections
        this.showSection('progressSection');
        this.hideSection('resultsSection');
        this.hideSection('errorSection');
        this.hideSection('liveUpdates');
        
        // Show/hide footers
        this.showSection('progressFooter');
        this.hideSection('resultsFooter');

        // Clear updates
        const updatesList = document.getElementById('updatesList');
        if (updatesList) updatesList.innerHTML = '';
        
        this.selectedUpdates.clear();
    }

    startProgressStream(streamUrl) {
        // Close existing stream
        if (this.eventSource) {
            this.eventSource.close();
        }

        // Start new EventSource stream
        this.eventSource = new EventSource(streamUrl);

        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleProgressUpdate(data);
            } catch (error) {
                console.error('Error parsing progress data:', error);
            }
        };

        this.eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            if (this.eventSource.readyState === EventSource.CLOSED) {
                this.handleStreamClosed();
            }
        };

        this.eventSource.onopen = () => {
            console.log('Progress stream connected');
        };
    }

    handleProgressUpdate(data) {
        if (data.type === 'keepalive') {
            return; // Ignore keepalive messages
        }

        const { stage, progress, message, job_id } = data;

        // Update progress bar
        this.updateProgressBar(progress, message);

        // Update stage information
        this.updateStageInfo(stage, message);

        // Handle stage-specific updates
        if (data.data && Object.keys(data.data).length > 0) {
            this.handleStageData(stage, data.data);
        }

        // Check if analysis is complete
        if (stage === 'completed') {
            this.handleAnalysisComplete(data.data);
        } else if (stage === 'error') {
            this.handleAnalysisError(message);
        } else if (stage === 'cancelled') {
            this.handleAnalysisCancelled();
        }
    }

    updateProgressBar(progress, message) {
        const progressBar = document.getElementById('analysisProgressBar');
        const progressText = document.getElementById('progressText');

        if (progressBar) {
            const percentage = Math.round(progress * 100);
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
        }

        if (progressText) {
            progressText.textContent = `${Math.round(progress * 100)}% - ${message}`;
        }
    }

    updateStageInfo(stage, message) {
        const currentStage = document.getElementById('currentStage');
        const stageDescription = document.getElementById('stageDescription');
        const stageIcon = document.getElementById('stageIcon');

        const stageInfo = this.getStageInfo(stage);

        if (currentStage) currentStage.textContent = stageInfo.title;
        if (stageDescription) stageDescription.textContent = message;
        if (stageIcon) stageIcon.innerHTML = stageInfo.icon;
    }

    getStageInfo(stage) {
        const stages = {
            'initializing': {
                title: 'Initializing Analysis',
                icon: '<i class="fas fa-play text-primary"></i>'
            },
            'extracting_content': {
                title: 'Extracting Content',
                icon: '<i class="fas fa-file-text text-info"></i>'
            },
            'detecting_entities': {
                title: 'Detecting Entities',
                icon: '<i class="fas fa-search text-warning"></i>'
            },
            'classifying_entities': {
                title: 'Classifying Entities',
                icon: '<i class="fas fa-tags text-primary"></i>'
            },
            'finding_relationships': {
                title: 'Finding Relationships',
                icon: '<i class="fas fa-project-diagram text-success"></i>'
            },
            'generating_updates': {
                title: 'Generating Updates',
                icon: '<i class="fas fa-magic text-purple"></i>'
            },
            'verifying_results': {
                title: 'Verifying Results',
                icon: '<i class="fas fa-check-circle text-info"></i>'
            },
            'finalizing': {
                title: 'Finalizing Analysis',
                icon: '<i class="fas fa-flag-checkered text-success"></i>'
            },
            'completed': {
                title: 'Analysis Complete',
                icon: '<i class="fas fa-check text-success"></i>'
            },
            'error': {
                title: 'Analysis Failed',
                icon: '<i class="fas fa-exclamation-triangle text-danger"></i>'
            },
            'cancelled': {
                title: 'Analysis Cancelled',
                icon: '<i class="fas fa-times text-warning"></i>'
            }
        };

        return stages[stage] || {
            title: 'Processing',
            icon: '<i class="fas fa-spinner fa-spin text-primary"></i>'
        };
    }

    handleStageData(stage, data) {
        // Show live updates if we have intermediate results
        if (stage === 'detecting_entities' && data.entities) {
            this.showLiveUpdate('Entities Detected', `Found ${data.entities.length} potential entities`);
        } else if (stage === 'classifying_entities' && data.classified) {
            this.showLiveUpdate('Entities Classified', `Classified ${data.classified} entities`);
        } else if (stage === 'finding_relationships' && data.relationships) {
            this.showLiveUpdate('Relationships Found', `Discovered ${data.relationships.length} relationships`);
        }
    }

    showLiveUpdate(title, description) {
        const liveUpdates = document.getElementById('liveUpdates');
        const updatesList = document.getElementById('updatesList');

        if (liveUpdates && updatesList) {
            liveUpdates.style.display = 'block';

            const updateItem = document.createElement('div');
            updateItem.className = 'list-group-item';
            updateItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h6 class="mb-1">${title}</h6>
                        <p class="mb-1">${description}</p>
                        <small class="text-muted">${new Date().toLocaleTimeString()}</small>
                    </div>
                    <span class="badge bg-primary rounded-pill">New</span>
                </div>
            `;

            updatesList.insertBefore(updateItem, updatesList.firstChild);

            // Limit to 5 updates
            while (updatesList.children.length > 5) {
                updatesList.removeChild(updatesList.lastChild);
            }
        }
    }

    handleAnalysisComplete(data) {
        this.isAnalyzing = false;
        
        // Update progress bar to complete
        const progressBar = document.getElementById('analysisProgressBar');
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.className = 'progress-bar bg-success';
        }

        // Close event source
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        // Show results
        if (data && data.result) {
            this.displayAnalysisResults(data.result);
        }
    }

    handleAnalysisError(errorMessage) {
        this.isAnalyzing = false;
        
        // Update progress bar to error
        const progressBar = document.getElementById('analysisProgressBar');
        if (progressBar) {
            progressBar.className = 'progress-bar bg-danger';
        }

        // Close event source
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        // Show error
        this.showSection('errorSection');
        this.hideSection('progressSection');
        
        const errorMessageEl = document.getElementById('errorMessage');
        if (errorMessageEl) {
            errorMessageEl.textContent = errorMessage;
        }
    }

    handleAnalysisCancelled() {
        this.isAnalyzing = false;
        
        // Update progress bar to cancelled
        const progressBar = document.getElementById('analysisProgressBar');
        if (progressBar) {
            progressBar.className = 'progress-bar bg-warning';
        }

        // Close event source
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('streamingCrossReferenceModal'));
        if (modal) {
            modal.hide();
        }
    }

    async cancelAnalysis() {
        if (!this.currentJobId || !this.isAnalyzing) {
            return;
        }

        try {
            const button = document.getElementById('cancelAnalysisBtn');
            const originalText = button.innerHTML;
            
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Cancelling...';

            // Get novel ID from current page
            const novelId = this.getCurrentNovelId();
            
            const response = await fetch(`/novel/${novelId}/cross-reference/job/${this.currentJobId}/cancel`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const result = await response.json();

            if (result.success && result.cancelled) {
                // Cancellation will be handled by the stream
                console.log('Analysis cancelled successfully');
            } else {
                this.showError('Failed to cancel analysis');
                button.disabled = false;
                button.innerHTML = originalText;
            }

        } catch (error) {
            console.error('Cancel analysis error:', error);
            this.showError('Failed to cancel analysis');
        }
    }

    displayAnalysisResults(analysis) {
        // Hide progress section, show results
        this.hideSection('progressSection');
        this.showSection('resultsSection');
        this.showSection('resultsFooter');
        this.hideSection('progressFooter');

        // Use existing cross-reference display logic
        const resultsContainer = document.getElementById('analysisResults');
        if (resultsContainer && window.crossReferenceManager) {
            // Set the analysis data and display results
            window.crossReferenceManager.currentAnalysis = analysis;
            resultsContainer.innerHTML = window.crossReferenceManager.buildAnalysisResultsHTML(analysis);
            
            // Bind checkbox events
            window.crossReferenceManager.bindUpdateCheckboxes();
        }
    }

    async applySelectedUpdates() {
        if (window.crossReferenceManager) {
            await window.crossReferenceManager.applySelectedUpdates();
        }
    }

    getCurrentNovelId() {
        // Extract novel ID from current URL or page data
        const pathParts = window.location.pathname.split('/');
        const novelIndex = pathParts.indexOf('novel');
        return novelIndex !== -1 && pathParts[novelIndex + 1] ? pathParts[novelIndex + 1] : null;
    }

    showSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) section.style.display = 'block';
    }

    hideSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) section.style.display = 'none';
    }

    showError(message) {
        // Use existing error display logic
        if (window.crossReferenceManager) {
            window.crossReferenceManager.showError(message);
        } else {
            alert(message);
        }
    }

    handleStreamClosed() {
        if (this.isAnalyzing) {
            console.log('Stream closed unexpectedly');
            // Could show a reconnection message or error
        }
    }
}

// Initialize streaming manager
const streamingCrossReferenceManager = new StreamingCrossReferenceManager();
