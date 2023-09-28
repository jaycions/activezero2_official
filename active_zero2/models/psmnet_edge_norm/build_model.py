from active_zero2.models.psmnet_edge_norm.psmnet_3 import PSMNetEdgeNormal


def build_model(cfg):
    model = PSMNetEdgeNormal(
        min_disp=cfg.PSMNetEdgeNormal.MIN_DISP,
        max_disp=cfg.PSMNetEdgeNormal.MAX_DISP,
        num_disp=cfg.PSMNetEdgeNormal.NUM_DISP,
        set_zero=cfg.PSMNetEdgeNormal.SET_ZERO,
        dilation=cfg.PSMNetEdgeNormal.DILATION,
        epsilon=cfg.PSMNetEdgeNormal.EPSILON,
        grad_threshold=cfg.PSMNetEdgeNormal.GRAD_THRESHOLD,
        use_off=cfg.PSMNetEdgeNormal.USE_OFF,
        use_volume=cfg.PSMNetEdgeNormal.USE_VOLUME,
        edge_weight=cfg.PSMNetEdgeNormal.EDGE_WEIGHT,
    )
    return model
