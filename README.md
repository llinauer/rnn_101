# rnn_101

## An introduction to Recurrent Neural Networks

Since I first learned about Neural Networks, I was always afraid of Recurrent Neural Networks, or RNNs. 
Well, maybe not afraid, but I had a good deal of respect for them.
"Normal", so-called feed-forward Neural Networks are conceptionally easy. You put data in, 
the Neural Network processes it, feeds the signal forward through the layers and then something comes out.
Not so with RNNs. First of all, you can put a sequence of arbitrary length in them (How?). Then, they 
not only feed the signal forward, but also back to itself? What the .. ?

But what I learned over the years is that (not only) in Machine Learning, sometimes the best way of learning something,
is by doing something. So, in this repo, i am actually doing something with RNNs.

## The setup

The first thing you need, is a task that an RNN will actually help you with solving.
I came up with the following exercise:

![Alt text](imgs/task_description.jpg)

Sum all the digits in a sequene of numbers together and output the result.
Simple enough, right?
