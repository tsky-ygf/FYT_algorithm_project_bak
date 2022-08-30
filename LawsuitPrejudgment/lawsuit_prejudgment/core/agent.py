from enum import Enum


class ActionType(Enum):
    ASK_A_FACTOR = "ASK_A_FACTOR"
    ASK_A_CLAIM = "ASK_A_FACTOR"
    REPORT = "REPORT"


class CivilPrejudgmentAgent:

    def __init__(self, *args, **kwargs):
        # Dialog State Tracking
        self.context = kwargs

    def _recover_status(self):
        pass

    def _get_next_action(self) -> ActionType:
        """ Dialog Policy.决定下一次的动作:{ASK_A_FACTOR, ASK_A_CLAIM, REPORT}. """
        next_action = ActionType.ASK_A_FACTOR
        return next_action

    def __call__(self, *args, **kwargs):
        # Recover Dialogue Status
        self._recover_status()
        # Dialog Policy
        next_action = self._get_next_action()
        # do action
        if next_action == ActionType.ASK_A_FACTOR:
            return
        if next_action == ActionType.ASK_A_CLAIM:
            return
        if next_action == ActionType.REPORT:
            return
