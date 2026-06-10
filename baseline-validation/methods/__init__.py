from methods.nonparams import EnergyMethod, MahalanobisMethod, MSPMethod, ViMMethod
from methods.params import NPOSMethod, VOSMethod

METHODS = {
    MSPMethod.name: MSPMethod,
    EnergyMethod.name: EnergyMethod,
    MahalanobisMethod.name: MahalanobisMethod,
    ViMMethod.name: ViMMethod,
    VOSMethod.name: VOSMethod,
    NPOSMethod.name: NPOSMethod,
}

__all__ = [
    "EnergyMethod",
    "MahalanobisMethod",
    "METHODS",
    "MSPMethod",
    "NPOSMethod",
    "ViMMethod",
    "VOSMethod",
]
