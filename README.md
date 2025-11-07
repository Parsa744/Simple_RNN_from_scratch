Base RNN Class

torchRNN: a minimal RNN cell built from explicit linear layers

Manually concatenates input and hidden state

Computes hidden state via ReLU

Computes output via ReLU

RNN Variants

Built on top of the base RNN cell:

seq2oneRNN
-  Processes a sequence of inputs and outputs a single vector.

one2seqRNN
-  Takes a single input and generates a sequence of outputs.

one2oneRNN
-  Recurrently applies the same input for a fixed number of steps.

seq2seqRNN
-  Maps an input sequence to an output sequence (encoder-like behavior).


install:
  ``` pip install torch numpy ```

basic use:
 ```
  rnn = torchRNN(input_size=10, hidden_size=20, output_size=5)
  
  input = torch.randn(10, 1)
  hidden = torch.zeros(20, 1)
  
  output, next_hidden = rnn(input, hidden)
 ```
