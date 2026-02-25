import collections

from ara.utils import build_from_cfg
from ..registry import PROCESS


class Process(object):
    """Compose multiple process sequentially."""
    def __init__(self, processes, cfg):
        assert isinstance(processes, collections.abc.Sequence)
        self.processes = []
        for process in processes:
            if isinstance(process, dict):
                process = build_from_cfg(process, PROCESS, default_args=dict(cfg=cfg))
                self.processes.append(process)
            elif callable(process):
                self.processes.append(process)
            else:
                raise TypeError("process must be callable or a dict")

    def __call__(self, data):
        for t in self.processes:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.processes:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string