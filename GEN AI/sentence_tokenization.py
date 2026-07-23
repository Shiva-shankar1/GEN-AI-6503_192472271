import re
text = """Artificial Intelligence is transforming the world.
It is used in healthcare, agriculture, and education.
Python makes AI development easier! Do you like AI? Yes, it is powerful."""
sentences = re.split(r'(?<=[.!?])\s+', text)
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")