import dataclasses


@dataclasses.dataclass
class Result:
    results_dir: str
    window_size: int
    loss_function: str
    hyperparams: str
