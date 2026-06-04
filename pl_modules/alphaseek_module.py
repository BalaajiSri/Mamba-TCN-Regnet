# Renamed to pl_modules/hybrid_ssm_trader_module.py — this shim preserves backward compatibility.
from pl_modules.hybrid_ssm_trader_module import HybridSSMTraderModule  # noqa: F401
AlphaSeekSignalModule = HybridSSMTraderModule  # legacy alias
