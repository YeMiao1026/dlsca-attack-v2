# Importing these registers each architecture with src.models.registry via
# its @register decorator — without this, registry.get(name) would find nothing.
from src.models import cnn_best, cnn_light, resnet  # noqa: F401
