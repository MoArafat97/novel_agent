/**
 * Relationship Graphs JavaScript
 * 
 * Creates interactive network visualizations of entity relationships using D3.js with:
 * - Force-directed graph layout
 * - Interactive nodes and edges
 * - Zoom and pan capabilities
 * - Entity type color coding
 * - Relationship strength visualization
 */

class RelationshipGraphManager {
    constructor() {
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.width = 800;
        this.height = 600;
        this.colors = {
            'characters': '#3498db',
            'locations': '#e74c3c',
            'lore': '#f39c12',
            'default': '#95a5a6'
        };
        this.init();
    }

    init() {
        // Initialize when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            this.bindGraphEvents();
        });
    }

    bindGraphEvents() {
        // Bind relationship graph button clicks
        document.addEventListener('click', (e) => {
            if (e.target.matches('.show-relationship-graph-btn')) {
                this.handleShowRelationshipGraph(e);
            } else if (e.target.matches('.export-graph-btn')) {
                this.handleExportGraph(e);
            }
        });
    }

    async handleShowRelationshipGraph(event) {
        const button = event.target.closest('.show-relationship-graph-btn');
        const entityType = button.dataset.entityType;
        const entityId = button.dataset.entityId;
        const novelId = button.dataset.novelId;

        if (!entityType || !entityId || !novelId) {
            this.showError('Missing required data for relationship graph');
            return;
        }

        try {
            // Show loading state
            this.setButtonLoading(button, true);

            // Fetch relationship data
            const relationships = await this.fetchRelationshipData(novelId, entityType, entityId);
            
            // Show graph modal
            this.showRelationshipGraphModal(relationships, entityType, entityId);

        } catch (error) {
            console.error('Relationship graph error:', error);
            this.showError('Failed to load relationship graph');
        } finally {
            this.setButtonLoading(button, false);
        }
    }

    async fetchRelationshipData(novelId, entityType, entityId) {
        // For now, we'll create a mock relationship structure
        // In a real implementation, this would fetch from the backend
        
        // Get current entity data
        const currentEntity = {
            id: entityId,
            type: entityType,
            name: this.getCurrentEntityName()
        };

        // Mock related entities and relationships
        const mockData = this.generateMockRelationshipData(currentEntity, novelId);
        
        return mockData;
    }

    generateMockRelationshipData(centerEntity, novelId) {
        // This is a placeholder - in real implementation, this would come from the backend
        const nodes = [centerEntity];
        const links = [];

        // Add some mock related entities
        const relatedEntities = [
            { id: 'char_1', type: 'characters', name: 'Related Character 1' },
            { id: 'char_2', type: 'characters', name: 'Related Character 2' },
            { id: 'loc_1', type: 'locations', name: 'Related Location 1' },
            { id: 'lore_1', type: 'lore', name: 'Related Lore 1' }
        ];

        nodes.push(...relatedEntities);

        // Add mock relationships
        relatedEntities.forEach((entity, index) => {
            links.push({
                source: centerEntity.id,
                target: entity.id,
                type: 'related_to',
                strength: Math.random() * 0.8 + 0.2, // Random strength between 0.2 and 1.0
                description: `Relationship ${index + 1}`
            });
        });

        // Add some inter-relationships
        if (relatedEntities.length > 1) {
            links.push({
                source: relatedEntities[0].id,
                target: relatedEntities[1].id,
                type: 'connected_to',
                strength: 0.5,
                description: 'Secondary relationship'
            });
        }

        return { nodes, links };
    }

    showRelationshipGraphModal(data, centerEntityType, centerEntityId) {
        // Create or show graph modal
        let modal = document.getElementById('relationshipGraphModal');
        if (!modal) {
            modal = this.createRelationshipGraphModal();
            document.body.appendChild(modal);
        }

        // Clear previous graph
        this.clearGraph();

        // Set up data
        this.nodes = data.nodes.map(d => ({ ...d }));
        this.links = data.links.map(d => ({ ...d }));

        // Show modal
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();

        // Initialize graph after modal is shown
        modal.addEventListener('shown.bs.modal', () => {
            this.initializeGraph();
        }, { once: true });
    }

    createRelationshipGraphModal() {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'relationshipGraphModal';
        modal.tabIndex = -1;
        modal.innerHTML = `
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-project-diagram me-2"></i>
                            Entity Relationship Graph
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Graph Controls -->
                        <div class="row mb-3">
                            <div class="col-md-8">
                                <div class="btn-group" role="group">
                                    <button type="button" class="btn btn-outline-primary" id="centerGraphBtn">
                                        <i class="fas fa-crosshairs me-1"></i>Center
                                    </button>
                                    <button type="button" class="btn btn-outline-primary" id="resetZoomBtn">
                                        <i class="fas fa-search-minus me-1"></i>Reset Zoom
                                    </button>
                                    <button type="button" class="btn btn-outline-primary" id="pauseSimulationBtn">
                                        <i class="fas fa-pause me-1"></i>Pause
                                    </button>
                                </div>
                            </div>
                            <div class="col-md-4 text-end">
                                <button type="button" class="btn btn-outline-success export-graph-btn">
                                    <i class="fas fa-download me-1"></i>Export PNG
                                </button>
                            </div>
                        </div>

                        <!-- Legend -->
                        <div class="row mb-3">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-body py-2">
                                        <div class="row align-items-center">
                                            <div class="col-auto">
                                                <strong>Legend:</strong>
                                            </div>
                                            <div class="col-auto">
                                                <span class="badge" style="background-color: ${this.colors.characters}">Characters</span>
                                            </div>
                                            <div class="col-auto">
                                                <span class="badge" style="background-color: ${this.colors.locations}">Locations</span>
                                            </div>
                                            <div class="col-auto">
                                                <span class="badge" style="background-color: ${this.colors.lore}">Lore</span>
                                            </div>
                                            <div class="col-auto">
                                                <small class="text-muted">• Line thickness = relationship strength</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Graph Container -->
                        <div id="relationshipGraphContainer" class="border rounded" style="height: 500px; overflow: hidden;">
                            <!-- D3.js graph will be rendered here -->
                        </div>

                        <!-- Selected Node Info -->
                        <div id="selectedNodeInfo" class="mt-3" style="display: none;">
                            <div class="card">
                                <div class="card-body">
                                    <h6 id="selectedNodeName">Node Information</h6>
                                    <div id="selectedNodeDetails"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" id="analyzeSelectedBtn" disabled>
                            <i class="fas fa-search me-1"></i>Analyze Selected
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Bind control events
        modal.querySelector('#centerGraphBtn').addEventListener('click', () => this.centerGraph());
        modal.querySelector('#resetZoomBtn').addEventListener('click', () => this.resetZoom());
        modal.querySelector('#pauseSimulationBtn').addEventListener('click', () => this.toggleSimulation());
        modal.querySelector('#analyzeSelectedBtn').addEventListener('click', () => this.analyzeSelectedNode());

        return modal;
    }

    initializeGraph() {
        const container = document.getElementById('relationshipGraphContainer');
        if (!container) return;

        // Clear container
        container.innerHTML = '';

        // Set dimensions
        const rect = container.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;

        // Create SVG
        this.svg = d3.select(container)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svg.select('.graph-group').attr('transform', event.transform);
            });

        this.svg.call(zoom);

        // Create main group for graph elements
        const g = this.svg.append('g').attr('class', 'graph-group');

        // Create force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(30));

        // Create links
        const link = g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(this.links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => Math.sqrt(d.strength * 10));

        // Create nodes
        const node = g.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(this.nodes)
            .enter().append('circle')
            .attr('r', 20)
            .attr('fill', d => this.colors[d.type] || this.colors.default)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .call(this.createDragBehavior());

        // Add labels
        const label = g.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(this.nodes)
            .enter().append('text')
            .text(d => d.name)
            .attr('font-size', '12px')
            .attr('font-family', 'Arial, sans-serif')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .style('pointer-events', 'none')
            .style('fill', '#333');

        // Add tooltips
        node.append('title')
            .text(d => `${d.name} (${d.type})`);

        link.append('title')
            .text(d => `${d.type}: ${d.description || 'No description'}`);

        // Add click events
        node.on('click', (event, d) => this.handleNodeClick(event, d));

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }

    createDragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }

    handleNodeClick(event, node) {
        // Highlight selected node
        this.svg.selectAll('circle')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);

        d3.select(event.target)
            .attr('stroke', '#000')
            .attr('stroke-width', 4);

        // Show node information
        this.showNodeInfo(node);

        // Enable analyze button
        const analyzeBtn = document.getElementById('analyzeSelectedBtn');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.dataset.nodeId = node.id;
            analyzeBtn.dataset.nodeType = node.type;
        }
    }

    showNodeInfo(node) {
        const infoContainer = document.getElementById('selectedNodeInfo');
        const nameElement = document.getElementById('selectedNodeName');
        const detailsElement = document.getElementById('selectedNodeDetails');

        if (infoContainer && nameElement && detailsElement) {
            nameElement.textContent = `${node.name} (${node.type})`;
            
            // Find connected nodes
            const connections = this.links.filter(link => 
                link.source.id === node.id || link.target.id === node.id
            );

            detailsElement.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <strong>Entity Type:</strong> ${node.type}<br>
                        <strong>Connections:</strong> ${connections.length}
                    </div>
                    <div class="col-md-6">
                        <strong>Connected To:</strong><br>
                        ${connections.map(link => {
                            const connectedNode = link.source.id === node.id ? link.target : link.source;
                            return `• ${connectedNode.name} (${link.type})`;
                        }).join('<br>')}
                    </div>
                </div>
            `;

            infoContainer.style.display = 'block';
        }
    }

    centerGraph() {
        if (this.svg) {
            const zoom = d3.zoom().scaleExtent([0.1, 4]);
            this.svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity.translate(this.width / 2, this.height / 2).scale(1)
            );
        }
    }

    resetZoom() {
        if (this.svg) {
            const zoom = d3.zoom().scaleExtent([0.1, 4]);
            this.svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity
            );
        }
    }

    toggleSimulation() {
        const button = document.getElementById('pauseSimulationBtn');
        if (this.simulation) {
            if (this.simulation.alpha() > 0) {
                this.simulation.stop();
                button.innerHTML = '<i class="fas fa-play me-1"></i>Resume';
            } else {
                this.simulation.restart();
                button.innerHTML = '<i class="fas fa-pause me-1"></i>Pause';
            }
        }
    }

    analyzeSelectedNode() {
        const analyzeBtn = document.getElementById('analyzeSelectedBtn');
        const nodeId = analyzeBtn.dataset.nodeId;
        const nodeType = analyzeBtn.dataset.nodeType;

        if (nodeId && nodeType) {
            // Close graph modal and trigger cross-reference analysis
            const modal = bootstrap.Modal.getInstance(document.getElementById('relationshipGraphModal'));
            if (modal) {
                modal.hide();
            }

            // Trigger cross-reference analysis for the selected node
            // This would need to be implemented based on your routing structure
            window.location.href = `/novel/${this.getCurrentNovelId()}/${nodeType}/${nodeId}`;
        }
    }

    handleExportGraph() {
        if (!this.svg) return;

        try {
            // Create a canvas element
            const canvas = document.createElement('canvas');
            canvas.width = this.width;
            canvas.height = this.height;
            const context = canvas.getContext('2d');

            // Set white background
            context.fillStyle = 'white';
            context.fillRect(0, 0, this.width, this.height);

            // Convert SVG to image (simplified version)
            const svgData = new XMLSerializer().serializeToString(this.svg.node());
            const img = new Image();
            
            img.onload = () => {
                context.drawImage(img, 0, 0);
                
                // Download the image
                const link = document.createElement('a');
                link.download = 'relationship-graph.png';
                link.href = canvas.toDataURL();
                link.click();
            };

            img.src = 'data:image/svg+xml;base64,' + btoa(svgData);

        } catch (error) {
            console.error('Export failed:', error);
            this.showError('Failed to export graph');
        }
    }

    clearGraph() {
        if (this.svg) {
            this.svg.remove();
            this.svg = null;
        }
        if (this.simulation) {
            this.simulation.stop();
            this.simulation = null;
        }
    }

    getCurrentEntityName() {
        // Extract entity name from page title or heading
        const heading = document.querySelector('h1, h2, .entity-name');
        return heading ? heading.textContent.trim() : 'Current Entity';
    }

    getCurrentNovelId() {
        // Extract novel ID from current URL
        const pathParts = window.location.pathname.split('/');
        const novelIndex = pathParts.indexOf('novel');
        return novelIndex !== -1 && pathParts[novelIndex + 1] ? pathParts[novelIndex + 1] : null;
    }

    setButtonLoading(button, loading) {
        if (loading) {
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
        } else {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-project-diagram me-1"></i>Relationship Graph';
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

// Initialize relationship graph manager
const relationshipGraphManager = new RelationshipGraphManager();
