from src.liveness.fusion.score_fusion import ScoreFusion
from src.attendance.attendance_rule import AttendanceRule


fusion = ScoreFusion()
rule = AttendanceRule(threshold=0.6)


recognition_score = 0.85
liveness_score = 0.90
background_score = 0.80


final_score = fusion.calculate(
    recognition_score,
    liveness_score,
    background_score
)

eligible = rule.is_eligible(final_score)


print("Recognition Score:", recognition_score)
print("Liveness Score:", liveness_score)
print("Background Score:", background_score)
print("Final Score:", final_score)
print("Attendance Eligible:", eligible)