"SharpEye Exceptions"

class SharpEyeError(Exception):
    "Base exception for all errors"

class PresetNotFound(Exception):
    "Raised when a preset YAML file does not exists."

class PresetValidationException(Exception):
    "Raised when preset YAML fails Pydantic validation."

class InvalidImageError(Exception):
    "Raised when input image cannot be read or decoded."

class MetricError(Exception):
    "Raised when a metric plugin fails during computation"