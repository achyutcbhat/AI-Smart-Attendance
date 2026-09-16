from src.recognition.student_registration import StudentRegistration


registration = StudentRegistration()

student_code = input("Enter student code: ")
name = input("Enter student name: ")
email = input("Enter student email: ")

registration.register_student(
    student_code,
    name,
    email
)