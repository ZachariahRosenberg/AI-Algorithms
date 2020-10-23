import io, os, random, requests
import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import tensorflow_addons    as tfa
tf.disable_v2_behavior()

class TextGenerator():

    def __init__(self, text, n_batches=None, batch_size=32, seq_length=16, rnn_size=512, embed_dim=300, lr=.001, n_epochs=250):

        self.text = self.prepare_text(text)
        self.vocab_to_int, self.int_to_vocab = self.create_text_int_tables(self.text.split())
        self.int_text = [self.vocab_to_int[word] for word in self.text.split()]

        self.n_batches  = n_batches
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.rnn_size   = rnn_size
        self.embed_dim  = embed_dim
        self.lr         = lr
        self.n_epochs   = n_epochs

        self.model, self.input_text, self.targets, self.learning_rate, self.initial_state, \
        self.cost, self.final_state, self.train_op = self.create_rnn_model()

    def prepare_text(self, text):
        # Lower case and swap out punctuation in text
        return ''.join([self.punctuation_tokenizer(str(letter).lower()) for letter in text])

    def create_text_int_tables(self, text):
        '''
        Allows us to encode words as an integer, which is then fed into the RNN.
        '''

        vocab_to_int = {word: i for i, word in enumerate(set(text))}
        int_to_vocab = {i: word for i, word in enumerate(set(text))}

        return vocab_to_int, int_to_vocab

    def punctuation_tokenizer(self, letter):
        '''
        Encode punctuation marks as words
        '''

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
                '\r':' ||newline|| ',
                ':' :' ||colon|| ',
            }

        if letter in punctuation.keys():
            return punctuation[letter]
        return letter

    def reverse_punctuation_tokenizer(self, token):
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

    def create_batches(self, text, batch_size, seq_length):
        num_batches = len(text)//(batch_size * seq_length)
        trunc_data  = text[:num_batches * (batch_size * seq_length)]

        sequences_x = np.array([[text[i+j] for j in range(0, seq_length)] for i in range(0, len(trunc_data), seq_length)])
        sequences_y = np.roll(sequences_x, -1)

        batches_x = np.reshape(sequences_x, (num_batches, batch_size, seq_length), order='F')
        batches_y = np.reshape(sequences_y, (num_batches, batch_size, seq_length), order='F')

        batches = np.array(list(zip(batches_x, batches_y)))

        return batches

    def create_rnn_model(self):

        vocab_size = len(self.vocab_to_int)

        train_graph = tf.Graph()
        with train_graph.as_default():

            input_text       = tf.placeholder(tf.int32,   (None, None), name='input')
            targets          = tf.placeholder(tf.int32,   (None, None), name='targets')
            learning_rate    = tf.placeholder(tf.float64, (None)      , name='learning_rate')
            input_data_shape = tf.shape(input_text)

            # LSTM layer
            lstm = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=.8)
            cell = tf.nn.rnn_cell.MultiRNNCell([drop])

            initial_state = cell.zero_state(input_data_shape[0], tf.int32)
            initial_state = tf.identity(initial_state, 'initial_state')

            #embedding layer
            embed_matrix = tf.Variable(tf.random_uniform((vocab_size, self.embed_dim),-1,1))
            embedding    = tf.nn.embedding_lookup(embed_matrix, input_text)

            #Final RNN layer
            outputs, final_state = tf.nn.dynamic_rnn(cell, embedding, dtype=tf.float32)
            final_state = tf.identity(final_state, 'final_state')

            #Fully Connected Output Layer
            logits = tf.layers.dense(outputs, vocab_size, activation=None)

            # Probabilities for generating words
            probs = tf.nn.softmax(logits, name='softmax_probs')

            # Loss function
            cost = tfa.seq2seq.sequence_loss(logits, targets, tf.ones([input_data_shape[0], input_data_shape[1]]))

            # Optimizer
            optimizer = tf.train.AdamOptimizer(self.lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

        return train_graph, input_text, targets, learning_rate, initial_state, cost, final_state, train_op

    def train(self):

        batches = self.create_batches(self.int_text, self.batch_size, self.seq_length)
        if self.n_batches is not None: 
            batches = np.random.choice(batches, self.n_batches)

        # Training
        with tf.Session(graph=self.model) as sess:
            sess.run(tf.global_variables_initializer())

            for i in tqdm(range(self.n_epochs)):
                state = sess.run(self.initial_state, {self.input_text: batches[0][0]})

                for batch_i, (x, y) in enumerate(batches):
                    batch = {
                        self.input_text   : x,
                        self.targets      : y,
                        self.learning_rate: self.lr,
                        self.initial_state: state,
                    }
                    train_loss, state, _ = sess.run([self.cost, self.final_state, self.train_op], batch)

                    # Show every 500 batches
                    if (i * len(batches) + batch_i) % 500 == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(i,batch_i,len(batches),train_loss))

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, './save')

        return True

    def generate_text(self, prime_word='the', seq_length=50, gen_length=5000):

        with tf.Session(graph=self.model) as sess:

            if os.path.isfile('./save.meta'):
                # Load saved model
                loader = tf.train.import_meta_graph('./save.meta')
                loader.restore(sess, './save')

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
                dyn_input = [[self.vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
                dyn_seq_length = len(dyn_input[0])

                # Get Prediction
                probabilities, prev_state = sess.run([probs, final_state], {input_text: dyn_input, init_state: prev_state})

                pred_word = self.int_to_vocab[np.argmax(probabilities[0][dyn_seq_length-1])]

                gen_sentences.append(pred_word)

            # Remove tokens
            script = ' '.join([self.reverse_punctuation_tokenizer(word) for word in gen_sentences])

        return script


if __name__ == "__main__":

    # Portrait of Dorian Gray
    r = requests.get('https://www.gutenberg.org/cache/epub/174/pg174.txt')
    text = r.text

    text = text.replace(u'\ufeff', '')

    text_generator = TextGenerator(text, rnn_size=250, n_epochs=50)
    text_generator.train()
    script = text_generator.generate_text(gen_length=5000)

    print(script)