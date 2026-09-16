from datetime import date

from src.attendance.duplicate_checker import DuplicateChecker


checker = DuplicateChecker()

student_id = 1
today = date.today()

result = checker.is_already_marked(
    student_id,
    today
)

print("Student ID:", student_id)
print("Date:", today)
print("Already Marked:", result)