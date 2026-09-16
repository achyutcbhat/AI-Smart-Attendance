from src.utils.config import load_config


class AttendanceDecision:

    def __init__(self):

        config = load_config()

        self.threshold = config["fusion"]["threshold"]

    def decide(self, final_score):

        if final_score >= self.threshold:
            return "accept"

        return "reject"