






from polls.models import Choice, Question  # Import the model classes we just wrote.

# No questions are in the system yet.
Question.objects.all()


q = Question(question_text="What's new?", pub_date=timezone.now())


















