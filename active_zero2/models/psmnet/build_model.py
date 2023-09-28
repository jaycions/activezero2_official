from active_zero2.models.psmnet.psmnet_3 import PSMNet


def build_model(cfg):
    model = PSMNet(
        maxdisp=cfg.PSMNet.MAX_DISP,
    )
    return model
