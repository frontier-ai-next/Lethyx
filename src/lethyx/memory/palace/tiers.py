class Tier:
    ALWAYS_ON = "always_on"
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    TRANSIENT = "transient"

    _ORDER = {ALWAYS_ON: 0, HOT: 1, WARM: 2, COLD: 3, TRANSIENT: 4}

    @classmethod
    def priority(cls, tier: str) -> int:
        return cls._ORDER.get(tier, 99)


class Scope:
    CROSS_RUN_DURABLE = "cross_run_durable"
    RUN_SCOPED = "run_scoped"
    PHASE_SCOPED = "phase_scoped"
    SUBTASK_SCOPED = "subtask_scoped"
    TRANSIENT = "transient"
