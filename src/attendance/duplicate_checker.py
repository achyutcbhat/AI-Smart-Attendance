from src.database.database import get_connection


class DuplicateChecker:

    def is_already_marked(self, student_id, attendance_date):

        connection = get_connection()
        cursor = connection.cursor()

        query = """
        SELECT attendance_id
        FROM attendance
        WHERE student_id = %s
        AND attendance_date = %s
        """

        cursor.execute(
            query,
            (student_id, attendance_date)
        )

        result = cursor.fetchone()

        cursor.close()
        connection.close()

        return result is not None