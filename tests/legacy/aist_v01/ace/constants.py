"""Fixed seeds and clustering defaults for reproducible ACE runs."""

ACE_RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 5000

# Shape-extent descriptor of the structure clustering (aist.inertia.inertia_shape_extent_features); the
# inertia colours of the figures use the same settings.
ACE_EXTENT_NORMALIZATION = "partial"
ACE_EXTENT_BLEND = 0.1   # blend fraction for the shape correction (0 = sphere, 1 = full)
