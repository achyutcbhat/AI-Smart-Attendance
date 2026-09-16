from src.database.database import get_connection


def add_attendance(
    student_id,
    attendance_date,
    attendance_time,
    recognition_score,
    liveness_score,
    background_score
):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO attendance
        (
            student_id,
            attendance_date,
            attendance_time,
            recognition_score,
            liveness_score,
            background_score
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (
            int(student_id),
            attendance_date,
            attendance_time,
            float(recognition_score),
            float(liveness_score),
            float(background_score)
        )
    )

    connection.commit()

    cursor.close()
    connection.close()