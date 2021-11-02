class CIP(object):
    """Composable Interaction Primitive"""
    def __init__(self):
        super(CIP, self).__init__()

    def run_body(self):
        raise NotImplementedError

    def run_head(self, target_ee_pose):
        """Implements grasping controller """
        raise NotImplementedError

    def sample_init(self):
        raise NotImplementedError

    def sample_effect(self):
        raise NotImplementedError

    def success(self, state):
        raise NotImplementedError

    def failure(self, state):
        raise NotImplementedError

    def check_safety(self, state):
        raise NotImplementedError

    def run(self, state):
        raise NotImplementedError

class Head(object):
    """docstring for Head."""
    def __init__(self, arg):
        super(Head, self).__init__()
        self.arg = arg

    def sample(self):
        raise NotImplementedError

    def optimize(self, data):
        raise NotImplementedError


class CIPAgent(object):
    """docstring for CIPAgent."""
    def __init__(self):
        super(CIPAgent, self).__init__()
        self.cips=[]
        self.cur_cip = None

    def plan(self, goal):
        raise NotImplementedError

    def get_action(state):
        raise NotImplementedError
