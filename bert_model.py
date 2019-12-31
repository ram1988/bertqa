import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
input_text = question + " [SEP] " + text
input_ids = tf.constant(tokenizer.encode(input_text, add_special_tokens=True))[None, :]  # Batch size 1
print(input_ids[0])
outputs = model(input_ids)
start_scores, end_scores = outputs[:2]
print(input_ids)

all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(all_tokens)
print(end_scores)
print(' '.join(all_tokens[tf.argmax(start_scores[0]) : tf.argmax(end_scores[0])+1]))