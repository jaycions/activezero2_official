from active_zero2.models.psmnet.build_model import build_model as build_psmnet
from active_zero2.models.psmnet_edge_norm.build_model import build_model as build_psmnetedgenorm

MODEL_LIST = (
    "PSMNet",
    "PSMNetEdgeNormal",
)


def build_model(cfg):
    if cfg.MODEL_TYPE == "PSMNet":
        model = build_psmnet(cfg)
    elif cfg.MODEL_TYPE == "PSMNetEdgeNormal":
        model = build_psmnetedgenorm(cfg)

    else:
        raise ValueError(f"Unexpected model type: {cfg.MODEL_TYPE}")

    return model
