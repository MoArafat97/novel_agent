"""
Confidence Calibration System

This module provides adaptive confidence threshold calibration for the cross-reference
system based on real usage data and performance metrics.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for confidence calibration."""
    total_classifications: int = 0
    correct_classifications: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    avg_confidence: float = 0.0
    confidence_accuracy_pairs: List[Tuple[float, bool]] = None
    
    def __post_init__(self):
        if self.confidence_accuracy_pairs is None:
            self.confidence_accuracy_pairs = []
    
    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        if self.total_classifications == 0:
            return 0.0
        return self.correct_classifications / self.total_classifications
    
    @property
    def precision(self) -> float:
        """Calculate precision (true positives / (true positives + false positives))."""
        true_positives = self.correct_classifications
        if true_positives + self.false_positives == 0:
            return 0.0
        return true_positives / (true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Calculate recall (true positives / (true positives + false negatives))."""
        true_positives = self.correct_classifications
        if true_positives + self.false_negatives == 0:
            return 0.0
        return true_positives / (true_positives + self.false_negatives)


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for different operations."""
    entity_acceptance: float = 0.6  # Minimum confidence to accept an entity
    premium_model_trigger: float = 0.8  # Use premium model if below this
    auto_approval: float = 0.9  # Auto-approve updates above this confidence
    rejection_threshold: float = 0.3  # Reject entities below this confidence
    batch_processing_min: float = 0.4  # Minimum confidence for batch processing
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ConfidenceThresholds':
        """Create from dictionary."""
        return cls(**data)


class ConfidenceCalibrator:
    """
    Adaptive confidence calibration system that learns from usage patterns
    and adjusts thresholds for optimal performance.
    """
    
    def __init__(self, 
                 data_dir: str = "data/calibration",
                 min_samples: int = 100,
                 calibration_interval: float = 3600.0):  # 1 hour
        """Initialize the confidence calibrator."""
        self.data_dir = Path(data_dir)
        self.min_samples = min_samples
        self.calibration_interval = calibration_interval
        
        # Current thresholds
        self.thresholds = ConfidenceThresholds()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.last_calibration = 0.0
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_calibration_data()
        
        logger.info(f"Initialized confidence calibrator with {self.metrics.total_classifications} historical samples")
    
    def record_classification(self, 
                            confidence: float, 
                            predicted_type: str, 
                            actual_type: str = None,
                            user_feedback: str = None) -> None:
        """
        Record a classification result for calibration.
        
        Args:
            confidence: Model confidence score
            predicted_type: Predicted entity type
            actual_type: Actual entity type (if known)
            user_feedback: User feedback ('correct', 'incorrect', 'uncertain')
        """
        self.metrics.total_classifications += 1
        
        # Determine if classification was correct
        is_correct = False
        if actual_type is not None:
            is_correct = predicted_type == actual_type
        elif user_feedback == 'correct':
            is_correct = True
        elif user_feedback == 'incorrect':
            is_correct = False
        else:
            # If no feedback, assume correct for high confidence, uncertain for low confidence
            is_correct = confidence > 0.7
        
        if is_correct:
            self.metrics.correct_classifications += 1
        
        # Record confidence-accuracy pair
        self.metrics.confidence_accuracy_pairs.append((confidence, is_correct))
        
        # Update average confidence
        self.metrics.avg_confidence = (
            (self.metrics.avg_confidence * (self.metrics.total_classifications - 1) + confidence) /
            self.metrics.total_classifications
        )
        
        # Trigger calibration if enough new data
        if (time.time() - self.last_calibration > self.calibration_interval and
            self.metrics.total_classifications >= self.min_samples):
            self._calibrate_thresholds()
    
    def get_optimal_threshold(self, target_metric: str = 'f1') -> float:
        """
        Calculate optimal threshold based on target metric.
        
        Args:
            target_metric: 'accuracy', 'precision', 'recall', or 'f1'
            
        Returns:
            Optimal threshold value
        """
        if len(self.metrics.confidence_accuracy_pairs) < self.min_samples:
            return self.thresholds.entity_acceptance
        
        # Sort by confidence
        sorted_pairs = sorted(self.metrics.confidence_accuracy_pairs)
        
        best_threshold = 0.5
        best_score = 0.0
        
        # Test different thresholds
        for i in range(10, len(sorted_pairs), 10):  # Sample every 10th point
            threshold = sorted_pairs[i][0]
            
            # Calculate metrics at this threshold
            tp = sum(1 for conf, correct in sorted_pairs if conf >= threshold and correct)
            fp = sum(1 for conf, correct in sorted_pairs if conf >= threshold and not correct)
            fn = sum(1 for conf, correct in sorted_pairs if conf < threshold and correct)
            
            if tp + fp == 0 or tp + fn == 0:
                continue
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = tp / len([p for p in sorted_pairs if p[0] >= threshold])
            
            if target_metric == 'precision':
                score = precision
            elif target_metric == 'recall':
                score = recall
            elif target_metric == 'accuracy':
                score = accuracy
            else:  # f1
                if precision + recall == 0:
                    score = 0
                else:
                    score = 2 * (precision * recall) / (precision + recall)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def _calibrate_thresholds(self) -> None:
        """Calibrate confidence thresholds based on performance data."""
        logger.info("Starting confidence threshold calibration")
        
        try:
            # Calculate optimal thresholds for different purposes
            optimal_acceptance = self.get_optimal_threshold('f1')
            optimal_precision = self.get_optimal_threshold('precision')
            optimal_recall = self.get_optimal_threshold('recall')
            
            # Update thresholds with conservative adjustments
            old_thresholds = self.thresholds.to_dict()
            
            # Entity acceptance: balance precision and recall
            self.thresholds.entity_acceptance = self._smooth_threshold_update(
                self.thresholds.entity_acceptance, optimal_acceptance, 0.1
            )
            
            # Premium model trigger: slightly higher than acceptance
            self.thresholds.premium_model_trigger = min(
                0.95, self.thresholds.entity_acceptance + 0.15
            )
            
            # Auto-approval: high precision threshold
            self.thresholds.auto_approval = self._smooth_threshold_update(
                self.thresholds.auto_approval, optimal_precision, 0.05
            )
            
            # Rejection threshold: lower than acceptance
            self.thresholds.rejection_threshold = max(
                0.1, self.thresholds.entity_acceptance - 0.3
            )
            
            # Batch processing: slightly lower than acceptance
            self.thresholds.batch_processing_min = max(
                0.2, self.thresholds.entity_acceptance - 0.2
            )
            
            # Log changes
            new_thresholds = self.thresholds.to_dict()
            changes = {k: f"{old_thresholds[k]:.3f} → {new_thresholds[k]:.3f}" 
                      for k in old_thresholds if abs(old_thresholds[k] - new_thresholds[k]) > 0.01}
            
            if changes:
                logger.info(f"Updated confidence thresholds: {changes}")
            else:
                logger.info("No significant threshold changes needed")
            
            self.last_calibration = time.time()
            self._save_calibration_data()
            
        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
    
    def _smooth_threshold_update(self, current: float, optimal: float, max_change: float) -> float:
        """Apply smooth threshold updates to avoid dramatic changes."""
        change = optimal - current
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
        
        new_threshold = current + change
        return max(0.1, min(0.99, new_threshold))  # Keep within reasonable bounds
    
    def _load_calibration_data(self) -> None:
        """Load calibration data from disk."""
        try:
            # Load thresholds
            thresholds_file = self.data_dir / "thresholds.json"
            if thresholds_file.exists():
                with open(thresholds_file, 'r') as f:
                    threshold_data = json.load(f)
                    self.thresholds = ConfidenceThresholds.from_dict(threshold_data)
            
            # Load metrics
            metrics_file = self.data_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics = PerformanceMetrics(**metrics_data)
            
        except Exception as e:
            logger.warning(f"Failed to load calibration data: {e}")
    
    def _save_calibration_data(self) -> None:
        """Save calibration data to disk."""
        try:
            # Save thresholds
            thresholds_file = self.data_dir / "thresholds.json"
            with open(thresholds_file, 'w') as f:
                json.dump(self.thresholds.to_dict(), f, indent=2)
            
            # Save metrics (limit size to prevent unbounded growth)
            metrics_data = asdict(self.metrics)
            if len(metrics_data['confidence_accuracy_pairs']) > 10000:
                # Keep only recent samples
                metrics_data['confidence_accuracy_pairs'] = metrics_data['confidence_accuracy_pairs'][-5000:]
            
            metrics_file = self.data_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
        except Exception as e:
            logger.warning(f"Failed to save calibration data: {e}")
    
    def get_current_thresholds(self) -> ConfidenceThresholds:
        """Get current confidence thresholds."""
        return self.thresholds
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_classifications': self.metrics.total_classifications,
            'accuracy': self.metrics.accuracy,
            'precision': self.metrics.precision,
            'recall': self.metrics.recall,
            'avg_confidence': self.metrics.avg_confidence,
            'current_thresholds': self.thresholds.to_dict(),
            'last_calibration': self.last_calibration
        }


# Global calibrator instance
_calibrator = None

def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get the global confidence calibrator instance."""
    global _calibrator
    if _calibrator is None:
        _calibrator = ConfidenceCalibrator()
    return _calibrator
