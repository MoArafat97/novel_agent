/**
 * Performance Optimizer for Frontend Operations
 * 
 * Provides performance improvements with:
 * - Request debouncing and throttling
 * - DOM update batching
 * - Memory leak prevention
 * - Caching for frequently accessed data
 * - Lazy loading for heavy operations
 */

class PerformanceOptimizer {
    constructor() {
        this.cache = new Map();
        this.pendingUpdates = new Set();
        this.requestQueue = new Map();
        this.observers = new Map();
        this.init();
    }

    init() {
        // Setup performance monitoring
        this.setupPerformanceMonitoring();
        
        // Setup intersection observer for lazy loading
        this.setupLazyLoading();
        
        // Setup mutation observer for DOM changes
        this.setupDOMObserver();
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    /**
     * Debounce function calls to prevent excessive API requests
     */
    debounce(func, wait, immediate = false) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    }

    /**
     * Throttle function calls to limit execution frequency
     */
    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    /**
     * Batch DOM updates to improve performance
     */
    batchDOMUpdates(updates) {
        return new Promise((resolve) => {
            requestAnimationFrame(() => {
                updates.forEach(update => {
                    if (typeof update === 'function') {
                        update();
                    }
                });
                resolve();
            });
        });
    }

    /**
     * Cache API responses to reduce server load
     */
    cacheResponse(key, data, ttl = 300000) { // 5 minutes default TTL
        const expiryTime = Date.now() + ttl;
        this.cache.set(key, {
            data,
            expiry: expiryTime
        });

        // Auto-cleanup expired entries
        setTimeout(() => {
            if (this.cache.has(key)) {
                const cached = this.cache.get(key);
                if (cached.expiry <= Date.now()) {
                    this.cache.delete(key);
                }
            }
        }, ttl);
    }

    /**
     * Get cached response if available and not expired
     */
    getCachedResponse(key) {
        if (this.cache.has(key)) {
            const cached = this.cache.get(key);
            if (cached.expiry > Date.now()) {
                return cached.data;
            } else {
                this.cache.delete(key);
            }
        }
        return null;
    }

    /**
     * Queue API requests to prevent duplicate calls
     */
    async queueRequest(key, requestFunc) {
        // If request is already in progress, wait for it
        if (this.requestQueue.has(key)) {
            return this.requestQueue.get(key);
        }

        // Check cache first
        const cached = this.getCachedResponse(key);
        if (cached) {
            return Promise.resolve(cached);
        }

        // Execute request and cache result
        const requestPromise = requestFunc()
            .then(result => {
                this.cacheResponse(key, result);
                this.requestQueue.delete(key);
                return result;
            })
            .catch(error => {
                this.requestQueue.delete(key);
                throw error;
            });

        this.requestQueue.set(key, requestPromise);
        return requestPromise;
    }

    /**
     * Setup lazy loading for images and heavy content
     */
    setupLazyLoading() {
        if ('IntersectionObserver' in window) {
            const lazyImageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.classList.remove('lazy');
                            lazyImageObserver.unobserve(img);
                        }
                    }
                });
            });

            // Observe all lazy images
            document.querySelectorAll('img[data-src]').forEach(img => {
                lazyImageObserver.observe(img);
            });

            this.observers.set('lazyImages', lazyImageObserver);
        }
    }

    /**
     * Setup DOM mutation observer for performance monitoring
     */
    setupDOMObserver() {
        if ('MutationObserver' in window) {
            const domObserver = new MutationObserver(
                this.throttle((mutations) => {
                    let hasSignificantChanges = false;
                    
                    mutations.forEach(mutation => {
                        if (mutation.type === 'childList' && mutation.addedNodes.length > 5) {
                            hasSignificantChanges = true;
                        }
                    });

                    if (hasSignificantChanges) {
                        this.optimizeDOMPerformance();
                    }
                }, 1000)
            );

            domObserver.observe(document.body, {
                childList: true,
                subtree: true,
                attributes: false
            });

            this.observers.set('domChanges', domObserver);
        }
    }

    /**
     * Optimize DOM performance after significant changes
     */
    optimizeDOMPerformance() {
        // Re-setup lazy loading for new images
        const newLazyImages = document.querySelectorAll('img[data-src]:not(.observed)');
        const lazyObserver = this.observers.get('lazyImages');
        
        if (lazyObserver) {
            newLazyImages.forEach(img => {
                img.classList.add('observed');
                lazyObserver.observe(img);
            });
        }

        // Cleanup unused event listeners
        this.cleanupEventListeners();
    }

    /**
     * Setup performance monitoring
     */
    setupPerformanceMonitoring() {
        if ('PerformanceObserver' in window) {
            // Monitor long tasks
            const longTaskObserver = new PerformanceObserver((list) => {
                list.getEntries().forEach(entry => {
                    if (entry.duration > 50) {
                        console.warn('Long task detected:', entry.duration + 'ms');
                    }
                });
            });

            try {
                longTaskObserver.observe({ entryTypes: ['longtask'] });
                this.observers.set('longTasks', longTaskObserver);
            } catch (e) {
                // Long task API not supported
            }

            // Monitor layout shifts
            const layoutShiftObserver = new PerformanceObserver((list) => {
                let cumulativeScore = 0;
                list.getEntries().forEach(entry => {
                    if (!entry.hadRecentInput) {
                        cumulativeScore += entry.value;
                    }
                });

                if (cumulativeScore > 0.1) {
                    console.warn('High cumulative layout shift:', cumulativeScore);
                }
            });

            try {
                layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
                this.observers.set('layoutShift', layoutShiftObserver);
            } catch (e) {
                // Layout shift API not supported
            }
        }
    }

    /**
     * Cleanup unused event listeners
     */
    cleanupEventListeners() {
        // Remove event listeners from removed elements
        const removedElements = document.querySelectorAll('[data-cleanup-listeners]');
        removedElements.forEach(element => {
            if (!document.contains(element)) {
                // Element was removed, cleanup its listeners
                element.removeEventListener('click', element._clickHandler);
                element.removeEventListener('change', element._changeHandler);
                // Add more cleanup as needed
            }
        });
    }

    /**
     * Preload critical resources
     */
    preloadCriticalResources() {
        const criticalResources = [
            '/static/js/api-utils.js',
            '/static/js/form-handler.js',
            '/static/css/style.css'
        ];

        criticalResources.forEach(resource => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.href = resource;
            link.as = resource.endsWith('.js') ? 'script' : 'style';
            document.head.appendChild(link);
        });
    }

    /**
     * Optimize images for better performance
     */
    optimizeImages() {
        const images = document.querySelectorAll('img:not([data-optimized])');
        images.forEach(img => {
            // Add loading="lazy" for better performance
            if (!img.hasAttribute('loading')) {
                img.loading = 'lazy';
            }

            // Add decode="async" for better performance
            if (!img.hasAttribute('decode')) {
                img.decode = 'async';
            }

            img.dataset.optimized = 'true';
        });
    }

    /**
     * Get performance metrics
     */
    getPerformanceMetrics() {
        if ('performance' in window) {
            const navigation = performance.getEntriesByType('navigation')[0];
            const paint = performance.getEntriesByType('paint');

            return {
                domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                firstPaint: paint.find(p => p.name === 'first-paint')?.startTime || 0,
                firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
                cacheSize: this.cache.size,
                activeRequests: this.requestQueue.size
            };
        }
        return null;
    }

    /**
     * Cleanup all observers and cached data
     */
    cleanup() {
        // Disconnect all observers
        this.observers.forEach(observer => {
            observer.disconnect();
        });
        this.observers.clear();

        // Clear cache
        this.cache.clear();

        // Clear request queue
        this.requestQueue.clear();

        console.log('Performance optimizer cleaned up');
    }
}

// Create global instance
window.performanceOptimizer = new PerformanceOptimizer();

// Optimize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.performanceOptimizer.optimizeImages();
    window.performanceOptimizer.preloadCriticalResources();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceOptimizer;
}
