# Define your PostgreSQL database connection settings
db_host = 'airbyte.cqqg4q5hnscs.ap-south-1.rds.amazonaws.com'
db_port = 5432
db_user = 'airbyte'
db_password = 'F648d&lTHltriVidRa0R'
db_name = 'learninganalytics'
schema_name = 'learninganalytics'
table_name = "student_data"
model_config_table = "academic_model_config"

fields = [
    ('Student_ID', 'Student id'), ('No_of_Logins', 'No. of logins'), ('ContentReads',
                                                                      'Content read'), ('ForumReads', 'Forum read'),
    ('ForumPosts', 'Forum posts'), ('Quiz_Reviews_before_submission',
                                    'No. of quiz review before submission'), ('Assignment1_delay', 'Assignment 1 delay'),
    ('Assignment2_delay', 'Assignment 2 delay'), ('Assignment3_delay',
                                                  'Assignment 3 delay'), ('Assignment1_submit', 'Assignment 1 submit time'),
    ('Assignment2_submit', 'Assignment 2 submit time'), ('Assignment3_submit',
                                                         'Assignment 3 submit time'),
    ('Average_time_to_submit_assignment',
     'Average time to submit assignments'), ('Engagement_Level', 'Engagement level'),
    ('assignment1_score', 'Assignment 1 score'), ('assignment2_score',
                                                  'Assignment 2 score'), ('assignment3_score', 'Assignment 3 score'),
    ('quiz1_score', 'Quiz 1 score'), ('Midterm_exam_score',
                                      'Midterm exam score'), ('final_exam_score', 'Final exam score'),
]