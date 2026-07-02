from .bc import BCAgent
from .calql import CalQLAgent
from .cql import CQLAgent
from .iql import IQLAgent
from .sac import SACAgent

agents = {
    "bc": BCAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
    "calql": CalQLAgent,
    "sac": SACAgent,
}
