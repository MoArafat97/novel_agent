/**
 * Stream Manager for EventSource Connections
 * 
 * Provides robust streaming with:
 * - Connection management
 * - Automatic reconnection
 * - Memory leak prevention
 * - Error handling
 * - Progress tracking
 */

class StreamManager {
    constructor() {
        this.activeStreams = new Map();
        this.reconnectAttempts = 3;
        this.reconnectDelay = 2000;
        this.maxReconnectDelay = 30000;
        this.init();
    }

    init() {
        // Cleanup streams on page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });

        // Handle visibility change to pause/resume streams
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseStreams();
            } else {
                this.resumeStreams();
            }
        });
    }

    /**
     * Create a new EventSource stream with enhanced error handling
     */
    createStream(streamId, url, options = {}) {
        // Close existing stream with same ID
        this.closeStream(streamId);

        const streamConfig = {
            url,
            options,
            eventSource: null,
            reconnectCount: 0,
            lastActivity: Date.now(),
            paused: false,
            callbacks: {
                onMessage: options.onMessage || (() => {}),
                onError: options.onError || (() => {}),
                onOpen: options.onOpen || (() => {}),
                onClose: options.onClose || (() => {})
            }
        };

        this.activeStreams.set(streamId, streamConfig);
        this.connectStream(streamId);

        return streamId;
    }

    /**
     * Connect or reconnect a stream
     */
    connectStream(streamId) {
        const config = this.activeStreams.get(streamId);
        if (!config || config.paused) return;

        try {
            config.eventSource = new EventSource(config.url);

            config.eventSource.onopen = (event) => {
                console.log(`Stream ${streamId} connected`);
                config.reconnectCount = 0;
                config.lastActivity = Date.now();
                config.callbacks.onOpen(event);
            };

            config.eventSource.onmessage = (event) => {
                config.lastActivity = Date.now();
                
                try {
                    const data = JSON.parse(event.data);
                    config.callbacks.onMessage(data, event);
                } catch (error) {
                    console.error(`Stream ${streamId} message parse error:`, error);
                    config.callbacks.onError(error);
                }
            };

            config.eventSource.onerror = (event) => {
                console.error(`Stream ${streamId} error:`, event);
                
                if (config.eventSource.readyState === EventSource.CLOSED) {
                    this.handleStreamClosed(streamId);
                } else {
                    config.callbacks.onError(event);
                }
            };

        } catch (error) {
            console.error(`Failed to create stream ${streamId}:`, error);
            config.callbacks.onError(error);
        }
    }

    /**
     * Handle stream closure and attempt reconnection
     */
    handleStreamClosed(streamId) {
        const config = this.activeStreams.get(streamId);
        if (!config) return;

        config.callbacks.onClose();

        // Attempt reconnection if within limits
        if (config.reconnectCount < this.reconnectAttempts) {
            config.reconnectCount++;
            const delay = Math.min(
                this.reconnectDelay * Math.pow(2, config.reconnectCount - 1),
                this.maxReconnectDelay
            );

            console.log(`Reconnecting stream ${streamId} in ${delay}ms (attempt ${config.reconnectCount})`);

            setTimeout(() => {
                if (this.activeStreams.has(streamId)) {
                    this.connectStream(streamId);
                }
            }, delay);
        } else {
            console.error(`Stream ${streamId} failed to reconnect after ${this.reconnectAttempts} attempts`);
            this.closeStream(streamId);
        }
    }

    /**
     * Close a specific stream
     */
    closeStream(streamId) {
        const config = this.activeStreams.get(streamId);
        if (config && config.eventSource) {
            config.eventSource.close();
            config.eventSource = null;
        }
        this.activeStreams.delete(streamId);
        console.log(`Stream ${streamId} closed`);
    }

    /**
     * Pause a stream (keeps config but closes connection)
     */
    pauseStream(streamId) {
        const config = this.activeStreams.get(streamId);
        if (config) {
            config.paused = true;
            if (config.eventSource) {
                config.eventSource.close();
                config.eventSource = null;
            }
        }
    }

    /**
     * Resume a paused stream
     */
    resumeStream(streamId) {
        const config = this.activeStreams.get(streamId);
        if (config && config.paused) {
            config.paused = false;
            this.connectStream(streamId);
        }
    }

    /**
     * Pause all active streams
     */
    pauseStreams() {
        this.activeStreams.forEach((config, streamId) => {
            this.pauseStream(streamId);
        });
    }

    /**
     * Resume all paused streams
     */
    resumeStreams() {
        this.activeStreams.forEach((config, streamId) => {
            this.resumeStream(streamId);
        });
    }

    /**
     * Get stream status
     */
    getStreamStatus(streamId) {
        const config = this.activeStreams.get(streamId);
        if (!config) return null;

        return {
            connected: config.eventSource && config.eventSource.readyState === EventSource.OPEN,
            paused: config.paused,
            reconnectCount: config.reconnectCount,
            lastActivity: config.lastActivity,
            timeSinceLastActivity: Date.now() - config.lastActivity
        };
    }

    /**
     * Check for stale streams and clean them up
     */
    checkStaleStreams() {
        const staleThreshold = 5 * 60 * 1000; // 5 minutes
        const now = Date.now();

        this.activeStreams.forEach((config, streamId) => {
            if (now - config.lastActivity > staleThreshold) {
                console.warn(`Closing stale stream ${streamId}`);
                this.closeStream(streamId);
            }
        });
    }

    /**
     * Send a message to a stream (for bidirectional communication)
     */
    async sendMessage(streamId, message) {
        const config = this.activeStreams.get(streamId);
        if (!config) {
            throw new Error(`Stream ${streamId} not found`);
        }

        // Extract base URL for sending messages
        const baseUrl = config.url.replace('/stream/', '/message/');
        
        try {
            const response = await window.apiUtils.postJSON(baseUrl, message);
            return response;
        } catch (error) {
            console.error(`Failed to send message to stream ${streamId}:`, error);
            throw error;
        }
    }

    /**
     * Get all active stream IDs
     */
    getActiveStreamIds() {
        return Array.from(this.activeStreams.keys());
    }

    /**
     * Get stream count
     */
    getStreamCount() {
        return this.activeStreams.size;
    }

    /**
     * Cleanup all streams
     */
    cleanup() {
        console.log('Cleaning up all streams...');
        this.activeStreams.forEach((config, streamId) => {
            this.closeStream(streamId);
        });
        this.activeStreams.clear();
    }

    /**
     * Start periodic maintenance
     */
    startMaintenance() {
        // Check for stale streams every 2 minutes
        setInterval(() => {
            this.checkStaleStreams();
        }, 2 * 60 * 1000);
    }
}

// Create global instance
window.streamManager = new StreamManager();

// Start maintenance
window.streamManager.startMaintenance();

// Enhanced cross-reference streaming with the new manager
class EnhancedStreamingCrossReference {
    constructor() {
        this.currentStreamId = null;
    }

    async startAnalysis(novelId, entityType, entityId) {
        try {
            // Start the analysis job
            const response = await window.apiUtils.postJSON(
                `/novel/${novelId}/cross-reference/analyze-stream`,
                { entity_type: entityType, entity_id: entityId }
            );

            if (!response.success) {
                throw new Error(response.error || 'Failed to start analysis');
            }

            // Create stream for progress updates
            this.currentStreamId = window.streamManager.createStream(
                `cross-ref-${Date.now()}`,
                response.stream_url,
                {
                    onMessage: (data) => this.handleProgressUpdate(data),
                    onError: (error) => this.handleStreamError(error),
                    onClose: () => this.handleStreamClose()
                }
            );

            return response.job_id;

        } catch (error) {
            console.error('Failed to start streaming analysis:', error);
            throw error;
        }
    }

    handleProgressUpdate(data) {
        // Handle progress updates (implement based on your UI needs)
        console.log('Progress update:', data);
    }

    handleStreamError(error) {
        console.error('Stream error:', error);
        window.apiUtils.showNotification('Stream connection error', 'warning');
    }

    handleStreamClose() {
        console.log('Stream closed');
        this.currentStreamId = null;
    }

    cancelAnalysis() {
        if (this.currentStreamId) {
            window.streamManager.closeStream(this.currentStreamId);
            this.currentStreamId = null;
        }
    }
}

// Create enhanced streaming instance
window.enhancedStreamingCrossReference = new EnhancedStreamingCrossReference();
