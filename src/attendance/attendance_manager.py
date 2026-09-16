from datetime import datetime

from src.attendance.attendance_rule import AttendanceRule
from src.attendance.duplicate_checker import DuplicateChecker
from src.database.attendance_repository import add_attendance


class AttendanceManager:

    def __init__(self):
        self.rule = AttendanceRule()
        self.duplicate_checker = DuplicateChecker()

    def mark_attendance(
        self,
        student_id,
        recognition_score,
        liveness_score,
        background_score,
        final_score
    ):

        if not self.rule.is_eligible(final_score):
            return "Rejected"

        now = datetime.now()

        if self.duplicate_checker.is_already_marked(
            student_id,
            now.date()
        ):
            return "Already Marked"

        add_attendance(
            student_id,
            now.date(),
            now.time(),
            recognition_score,
            liveness_score,
            background_score
        )

        return "Attendance Marked"