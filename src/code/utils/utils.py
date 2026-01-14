from .query_strategies import RandomSampling
from .query_strategies import EntropySampling
from .query_strategies import HybridSampling

# get strategies
def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "EntropySampling":
        return EntropySampling
    # elif name == "EntropySamplingDropout":
    #     return EntropySamplingDropout
    # elif name == "BALDDropout":
    #     return BALDDropout
    # elif name == "AdversarialAttack":
    #     return AdversarialAttack
    # elif name == "AdversarialAttack_efficient":
    #     return AdversarialAttack_efficient
    # elif name == "KCenterGreedy":
    #     return KCenterGreedy
    # elif name == "ClusterMarginSampling":
    #     return ClusterMarginSampling
    elif name == "HybridSampling":
        return HybridSampling
    else:
        raise NotImplementedError
