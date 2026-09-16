from src.utils.config import load_config


class AttendanceRule:

    def __init__(self):
        config = load_config()
        self.threshold = config["fusion"]["threshold"]

    def is_eligible(self, final_score):
        return final_score >= self.threshold