from src.attendance.attendance_rule import AttendanceRule


rule = AttendanceRule(threshold=0.6)

scores = [
    0.90,
    0.75,
    0.60,
    0.50,
    0.30
]

for score in scores:

    result = rule.is_eligible(score)

    print(
        f"Score: {score:.2f} "
        f"-> Eligible: {result}"
    )