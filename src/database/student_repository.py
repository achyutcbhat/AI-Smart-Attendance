from database.database import get_connection


def add_student(student_code, name, email=None, embedding_path=None):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO students
        (student_code, name, email, embedding_path)
        VALUES (%s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (student_code, name, email, embedding_path)
    )

    connection.commit()

    cursor.close()
    connection.close()