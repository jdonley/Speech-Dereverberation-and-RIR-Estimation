import yaml
import torch as t

def getConfig(config_path="./configs/config.yaml"):
    config = {}
    with open(config_path, "r") as cfgFile:
        try:
            config = yaml.safe_load(cfgFile)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def getTestConfig():
    return getConfig("./configs/test_config.yaml")

# def unwrap(p, discont=None, dim=-1, *, period=2*t.pi):
#     nd = p.ndim
#     dd = t.diff(p, dim=dim)
#     if discont is None:
#         discont = period/2
#     slice1 = [slice(None, None)]*nd     # full slices
#     slice1[dim] = slice(1, None)
#     slice1 = tuple(slice1)
#     dtype = t.result_type(dd, period)
#     if dtype == t.int:
#         interval_high, rem = divmod(period, 2)
#         boundary_ambiguous = rem == 0
#     else:
#         interval_high = period / 2
#         boundary_ambiguous = True
#     interval_low = -interval_high
#     ddmod = (dd - interval_low) % (period) + interval_low
#     if boundary_ambiguous:
#         # for `mask = (abs(dd) == period/2)`, the above line made
#         # `ddmod[mask] == -period/2`. correct these such that
#         # `ddmod[mask] == sign(dd[mask])*period/2`.
#         _nx.copyto(ddmod, interval_high,
#                    where=(ddmod == interval_low) & (dd > 0))
#     ph_correct = ddmod - dd
#     _nx.copyto(ph_correct, 0, where=abs(dd) < discont)
#     up = array(p, copy=True, dtype=dtype)
#     up[slice1] = p[slice1] + ph_correct.cumsum(axis)
#     return up