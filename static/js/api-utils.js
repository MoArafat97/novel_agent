/**
 * API Utilities for Frontend-Backend Communication
 * 
 * Provides centralized API handling with:
 * - Consistent error handling
 * - Request timeout management
 * - Response validation
 * - Loading state management
 * - Retry logic for failed requests
 */

class APIUtils {
    constructor() {
        this.defaultTimeout = 30000; // 30 seconds
        this.retryAttempts = 3;
        this.retryDelay = 1000; // 1 second
    }

    /**
     * Enhanced fetch with timeout, retry, and error handling
     */
    async fetchWithRetry(url, options = {}, retries = this.retryAttempts) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), options.timeout || this.defaultTimeout);

        const fetchOptions = {
            ...options,
            signal: controller.signal
        };

        try {
            const response = await fetch(url, fetchOptions);
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return response;
        } catch (error) {
            clearTimeout(timeoutId);

            if (retries > 0 && !controller.signal.aborted) {
                console.warn(`Request failed, retrying... (${retries} attempts left)`, error);
                await this.delay(this.retryDelay);
                return this.fetchWithRetry(url, options, retries - 1);
            }

            throw error;
        }
    }

    /**
     * POST request with JSON data
     */
    async postJSON(url, data, options = {}) {
        const response = await this.fetchWithRetry(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            body: JSON.stringify(data),
            ...options
        });

        return this.parseResponse(response);
    }

    /**
     * POST request with form data
     */
    async postForm(url, formData, options = {}) {
        const response = await this.fetchWithRetry(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                ...options.headers
            },
            body: formData,
            ...options
        });

        return this.parseResponse(response);
    }

    /**
     * GET request
     */
    async get(url, options = {}) {
        const response = await this.fetchWithRetry(url, {
            method: 'GET',
            ...options
        });

        return this.parseResponse(response);
    }

    /**
     * Parse and validate API response
     */
    async parseResponse(response) {
        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            
            // Validate response structure
            if (typeof data !== 'object') {
                throw new Error('Invalid response format');
            }

            return data;
        } else {
            const text = await response.text();
            return { success: true, data: text };
        }
    }

    /**
     * Utility delay function
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Show user-friendly error messages
     */
    handleError(error, context = '') {
        console.error(`API Error ${context}:`, error);

        let message = 'An unexpected error occurred. Please try again.';

        if (error.name === 'AbortError') {
            message = 'Request timed out. Please check your connection and try again.';
        } else if (error.message.includes('HTTP 404')) {
            message = 'The requested resource was not found.';
        } else if (error.message.includes('HTTP 403')) {
            message = 'You do not have permission to perform this action.';
        } else if (error.message.includes('HTTP 500')) {
            message = 'Server error. Please try again later.';
        } else if (error.message.includes('Failed to fetch')) {
            message = 'Network error. Please check your connection.';
        } else if (error.message) {
            message = error.message;
        }

        return message;
    }

    /**
     * Show loading state on button
     */
    setButtonLoading(button, loading, originalText = null) {
        if (loading) {
            if (!button.dataset.originalText) {
                button.dataset.originalText = button.innerHTML;
            }
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
        } else {
            button.disabled = false;
            button.innerHTML = originalText || button.dataset.originalText || 'Submit';
            delete button.dataset.originalText;
        }
    }

    /**
     * Show notification to user
     */
    showNotification(message, type = 'info', duration = 5000) {
        // Try to use existing notification system
        if (window.showNotification) {
            window.showNotification(message, type);
            return;
        }

        // Fallback to creating our own notification
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        // Auto-remove after duration
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, duration);
    }

    /**
     * Debounce function to prevent rapid API calls
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Validate form data before sending
     */
    validateFormData(data, requiredFields = []) {
        const errors = [];

        for (const field of requiredFields) {
            if (!data[field] || (typeof data[field] === 'string' && !data[field].trim())) {
                errors.push(`${field} is required`);
            }
        }

        return {
            isValid: errors.length === 0,
            errors
        };
    }

    /**
     * Clean up event sources and abort controllers
     */
    cleanup() {
        // This can be extended to track and cleanup resources
        console.log('API Utils cleanup called');
    }
}

// Create global instance
window.apiUtils = new APIUtils();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.apiUtils) {
        window.apiUtils.cleanup();
    }
});
