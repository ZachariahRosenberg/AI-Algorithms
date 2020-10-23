import io, random
import numpy as np, tensorflow as tf
from tqdm               import tqdm
from tensorflow.contrib import seq2seq


def create_lookup_tables(text):

    vocab_to_int = {word: i for i, word in enumerate(set(text))}
    int_to_vocab = {i: word for i, word in enumerate(set(text))}

    return vocab_to_int, int_to_vocab

def token_swap(letter):

    punctuation = {
            '.' :' ||period|| ',
            ',' :' ||comma|| ',
            '"' :' ||quotation|| ',
            ';' :' ||semicolon|| ',
            '!' :' ||exclamation|| ',
            '?' :' ||question|| ',
            '(' :' ||l-parentheses|| ',
            ')' :' ||r-parentheses|| ',
            '--':' ||dashdash|| ',
            '\n':' ||newline|| ',
            ':' :' ||colon|| ',
           }

    if letter in punctuation.keys():
    	return punctuation[letter]
    return letter

def r_token_swap(token):

    punctuation = {
            '||period||'        : '.',
            '||comma||'         : ',',
            '||quotation||'     : '"',
            '||semicolon||'     : ';',
            '||exclamation||'   : '!',
            '||question||'      : '?',
            '||l-parentheses||' : '(',
            '||r-parentheses||' : ')',
            '||dashdash||'      : '--',
            '||newline||'       : '\n',
            '||colon||'         : ':'
           }

    if token in punctuation.keys():
    	return punctuation[token]
    return token

def get_batches(int_text, batch_size, seq_length):

    num_batches = len(int_text)//(batch_size * seq_length)
    trunc_data  = int_text[:num_batches * (batch_size * seq_length)]
    
    sequences_x = np.array([[int_text[i+j] for j in range(0,seq_length)] for i in range(0, len(trunc_data), seq_length)])
    sequences_y = np.roll(sequences_x, -1)
    
    batches_x = np.reshape(sequences_x, (num_batches, batch_size, seq_length), order='F')
    batches_y = np.reshape(sequences_y, (num_batches, batch_size, seq_length), order='F')
    
    batches = np.array(list(zip(batches_x, batches_y)))

    return batches

def train_rnn(int_text, vocab_size, n_batches=None, batch_size=32, seq_length=16, rnn_size=512, embed_dim=300, lr=.001, num_epochs=250, save_dir='./save'):

	batches = get_batches(int_text, batch_size, seq_length)

	if n_batches is not None:
		batches = np.random.choice(batches, n_batches)

	train_graph = tf.Graph()
	with train_graph.as_default():

		input_text    = tf.placeholder(tf.int32,   (None, None), name='input')
		targets       = tf.placeholder(tf.int32,   (None, None), name='targets')
		learning_rate = tf.placeholder(tf.float64, (None)      , name='learning_rate')
		input_data_shape = tf.shape(input_text)

		# RNN cell
		lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
		drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=.8)
		cell = tf.contrib.rnn.MultiRNNCell([drop])

		initial_state = cell.zero_state(input_data_shape[0], tf.int32)
		initial_state = tf.identity(initial_state, 'initial_state')

		#embedding layer
		embed_matrix = tf.Variable(tf.random_uniform((vocab_size, embed_dim),-1,1))
		embedding    = tf.nn.embedding_lookup(embed_matrix, input_text)

		#LSTM layer
		outputs, final_state = tf.nn.dynamic_rnn(cell, embedding, dtype=tf.float32)
		final_state = tf.identity(final_state, 'final_state')

		#Fully Connected Layer
		logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)

		# Probabilities for generating words
		probs = tf.nn.softmax(logits, name='softmax_probs')

		# Loss function
		cost = seq2seq.sequence_loss(logits, targets, tf.ones([input_data_shape[0], input_data_shape[1]]))

		# Optimizer
		optimizer = tf.train.AdamOptimizer(lr)

		# Gradient Clipping
		gradients = optimizer.compute_gradients(cost)
		capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
		train_op = optimizer.apply_gradients(capped_gradients)

	# Training
	with tf.Session(graph=train_graph) as sess:
		sess.run(tf.global_variables_initializer())

		for i in tqdm(range(num_epochs)):
			state = sess.run(initial_state, {input_text: batches[0][0]})

			for batch_i, (x, y) in enumerate(batches):
				feed = {
				    input_text: x,
				    targets: y,
				    initial_state: state,
				    learning_rate: lr}
				train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

				# Show every 500 batches
				if (i * len(batches) + batch_i) % 500 == 0:
					print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(i,batch_i,len(batches),train_loss))

		# Save Model
		if save_dir is None:
			save_dir = './save'
		saver = tf.train.Saver()
		saver.save(sess, save_dir)

	return train_graph

def generate(int_text, int_to_vocab, vocab_to_int, graph=None, prime_word='map', seq_length=50, gen_length=5000, load_dir='./save'):

	with tf.Session(graph=graph) as sess:

		if load_dir is not None:
			# Load saved model
			loader = tf.train.import_meta_graph(load_dir + '.meta')
			loader.restore(sess, load_dir)

		# Get Tensors from loaded model
		input_text  = tf.get_default_graph().get_tensor_by_name('input:0')
		init_state  = tf.get_default_graph().get_tensor_by_name('initial_state:0')
		final_state = tf.get_default_graph().get_tensor_by_name('final_state:0')
		probs       = tf.get_default_graph().get_tensor_by_name('softmax_probs:0')

		# Sentences generation setup
		gen_sentences = [prime_word]
		prev_state = sess.run(init_state, {input_text: [[1]]})

		# Generate sentences
		for n in range(gen_length):
			# Dynamic Input
			dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
			dyn_seq_length = len(dyn_input[0])

			# Get Prediction
			probabilities, prev_state = sess.run([probs, final_state], {input_text: dyn_input, init_state: prev_state})

			pred_word = int_to_vocab[np.argmax(probabilities[0][dyn_seq_length-1])]

			gen_sentences.append(pred_word)
	    
		# Remove tokens
		script = []

		for word in gen_sentences:
			script.append(r_token_swap(word))
		script = ' '.join(script)

	return script

def demo(text_file=None):
	
	if text_file is None:
		text_file = 'genesis.txt'
	text = open(text_file, 'r+').read()


	letters  = []
	for letter in text:
		letter = token_swap(str(letter).lower())
		letters.append(letter)
	text = ''.join(letters)

	vocab_to_int, int_to_vocab = create_lookup_tables(text.split())

	int_text = []
	for word in text.split():
		int_text.append(vocab_to_int[word])
	print(int_to_vocab)
	graph  = train_rnn(int_text, len(vocab_to_int))
	script = generate(int_text, int_to_vocab, vocab_to_int, prime_word=int_to_vocab[0])

	return script


if __name__ == "__main__":
    script = demo('HUD.txt')
    print(script)