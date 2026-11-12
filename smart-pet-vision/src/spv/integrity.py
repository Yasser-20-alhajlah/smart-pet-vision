"""Strict data/evaluation contracts; no claim of verified animal identities."""
import numpy as np


def validate_metadata(df):
    required = {'image', 'label', 'species', 'breed', 'split', 'group', 'md5'}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError('Metadata must be nonempty and include split/group/hash/label fields.')
    if df[list(required)].isna().any().any():
        raise ValueError('Missing metadata cannot pass a leakage audit.')
    if df.image.astype(str).str.strip().eq('').any() or df.image.duplicated().any():
        raise ValueError('Image IDs must be unique and nonempty.')
    if df.md5.astype(str).str.strip().eq('').any():
        raise ValueError('Empty hashes cannot establish duplicate separation.')
    if not df.split.isin(['train', 'val', 'test']).all() or set(df.split) != {'train', 'val', 'test'}:
        raise ValueError('Require nonempty train, val and test partitions.')
    if not df.species.isin(['cat', 'dog']).all() or not df.label.isin([0, 1]).all():
        raise ValueError('Only cat=0 and dog=1 are valid classification targets.')
    if not (df.species.map({'cat':0, 'dog':1}) == df.label).all():
        raise ValueError('Species and numeric labels disagree.')
    if (df.groupby('group').split.nunique() > 1).any() or (df.groupby('md5').split.nunique() > 1).any():
        raise ValueError('Duplicate group or image hash crosses partitions.')
    for key in ('animal_group', 'source_group'):
        if key in df:
            if df[key].isna().any() or df[key].astype(str).str.strip().eq('').any():
                raise ValueError(f'{key} is incomplete.')
            if (df.groupby(key).split.nunique() > 1).any():
                raise ValueError(f'{key} crosses partitions.')


def validate_predictions(probs, labels, meta):
    probs, labels = np.asarray(probs), np.asarray(labels)
    if (probs.ndim != 2 or probs.shape[1] != 2 or len(probs) == 0
            or labels.shape != (len(probs),) or len(meta) != len(probs)
            or not np.isfinite(probs).all() or not np.isin(labels, [0,1]).all()
            or ((probs < 0) | (probs > 1)).any()
            or not np.allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Evaluation requires aligned, finite binary probabilities and labels.')
    if not {'image', 'label', 'species', 'breed'}.issubset(meta.columns):
        raise ValueError('Prediction metadata requires image, label, species and breed.')
    if meta.image.isna().any() or meta.image.duplicated().any():
        raise ValueError('Prediction image IDs must be complete and unique.')
    if not np.array_equal(meta.label.to_numpy(), labels):
        raise ValueError('Prediction order/labels differ from metadata.')
    if not np.array_equal(meta.species.map({'cat':0,'dog':1}).to_numpy(), labels):
        raise ValueError('Prediction species disagree with labels.')


def training_class_weights(labels, strategy='none'):
    """Optional new experiment, fitted ONLY on training labels. Default is historical loss."""
    if strategy not in ('none', 'balanced'):
        raise ValueError('class_balance must be none or balanced.')
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0,1]).all():
        raise ValueError('Training labels must be nonempty binary targets.')
    if strategy == 'none':
        return None
    counts = np.bincount(labels.astype(int), minlength=2)
    if (counts == 0).any():
        raise ValueError('Balanced training requires both species in the training partition.')
    return len(labels)/(2*counts)
