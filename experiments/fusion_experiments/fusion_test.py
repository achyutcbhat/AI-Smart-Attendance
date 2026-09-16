from src.liveness.fusion.score_fusion import ScoreFusion
from src.liveness.fusion.decision import AttendanceDecision

fusion = ScoreFusion()
decision = AttendanceDecision()

recognition_score = 0.85
liveness_score = 0.90
background_score = 0.80

final_score = fusion.calculate(
    recognition_score,
    liveness_score,
    background_score
)

result = decision.decide(final_score)

print("Score Fusion Experiment")
print("Recognition Score:", recognition_score)
print("Liveness Score:", liveness_score)
print("Background Score:", background_score)
print("Final Score:", final_score)
print("Decision:", result)