from .q_learner import QLearner
from .offpg_learner import OffPGLearner
from .cont_offpg_learner import ContOffPGLearner
from .cooperative_maddpg_learner import CoopMADDPGLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["cont_offpg_learner"] = ContOffPGLearner
REGISTRY["coop_maddpg_learner"] = CoopMADDPGLearner
