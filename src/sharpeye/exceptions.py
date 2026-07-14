"""SharpEye custom exceptions."""


class SharpEyeError(Exception):
    """Base exception for all SharpEye errors."""


class PresetNotFoundError(SharpEyeError):
    """Raised when a preset YAML file does not exist."""


class PresetValidationError(SharpEyeError):
    """Raised when preset YAML fails Pydantic validation."""


class InvalidImageError(SharpEyeError):
    """Raised when input image cannot be read or decoded."""


class MetricError(SharpEyeError):
    """Raised when a metric plugin fails during computation."""