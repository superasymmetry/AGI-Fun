# determines whether the question is concept-based or not.
contains_concept_prompt = '''A concept is a compressible rule system that allows you to generate or classify many specific instances from a small number of underlying principles. Concepts are generative categories: once you know the rules, you can recognize or produce valid examples across many situations. Good concepts are transferable, often taught with a name alone, and clarify reasoning by compressing complexity into simple, repeatable logic. Examples include the sunk cost fallacy, Pareto optimality, and a haiku. 

Not all terms or ideas are concepts. Some may be real or useful, but are not compressible into simple rules that can be flexibly applied. Examples of non-concepts include names of research areas (ie. graph theory), broad statistical findings, domain-specific knowledge, or complex mechanisms that resist clean summarization. Examples: gene flow, GDP trends, the Industrial Revolution.

To answer the following question correctly, must the person answering the question understand a specific concept correctly? Use as much text as needed to respond. At the end, write “ANSWER:” followed by either “Yes” or “No”. On the next line, write “CONCEPT:”, followed by the name of the single concept if there is one. Include no other text.'''

subquestion_generation_prompt = '''Your goal is to test concept understanding in many different ways. Do not just rephrase the same question. Be creative. Ask at least one question from the following categories:
(1) Write an instance of the concept. Ask the model to classify the instance. If you were testing the concept of a haiku, for example, you could write a haiku and ask for a classification of your poem.
(2) Write an example that is NOT an instance of the concept. Ask the model to classify the instance.
(3) Write an example that is close but still NOT an instance of the concept. Ask the model to classify the instance.
(4) Generate an example of the concept, given a creative set of constraints. For example, you might ask the model to generate an instance of the sunk cost fallacy that includes the word "banana" and takes place at the movie theater.
(5) Generate something that it NOT an example of the concept, given a creative set of constraints. For example, you might ask the model to generate an instance of Pareto optimality, where only one player has a Pareto optimal strategy.
(6) Edit some positive instance of a concept so that it no longer qualifies as that concept. An example of this would be to take a haiku and change one of the lines so that it no longer fits the 5-7-5 syllable structure.
(7) Edit some negative instance of a concept so that it now qualifies as that concept.
(8) Mask out part of a negative instance of the concept and ask for the text that could replace the blank to make the result a true instance of the concept.
(9) Mask out part of a positive instance of the concept and ask for the text that could replace the blank to make the result a false instance of the concept.

Do not include any answers, just questions. The questions should all require long answers. They should have never been seen before and should not be similar to anything in your training data, so the only way to get them right is to actually know the concept. 

You can reason all you'd like, but please put every question between the tags <question> and </question>.
'''