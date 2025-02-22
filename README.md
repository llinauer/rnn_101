# rnn_101

## An introduction to Recurrent Neural Networks

Since I first learned about Neural Networks, I was always afraid of Recurrent Neural Networks, or RNNs. 
Well, maybe not afraid, but I had a good deal of respect for them.
"Normal", so-called feed-forward Neural Networks are conceptually easy. You put data in, 
the Neural Network processes it, feeds the signal forward through the layers and then something comes out.
Not so with RNNs. First of all, you can put a sequence of arbitrary length in them (How?). Then, they 
not only feed the signal forward, but also back to itself? What the .. ?

But what I learned over the years is that (not only) in Machine Learning, sometimes the best way of learning something,
is by doing something. So, in this repo, i am actually doing something with RNNs.

## The setup

The first thing you need, is a task that an RNN will actually help you with solving.
I came up with the following exercise:

![Alt text](imgs/task_description.jpg)

Sum all the digits in a sequence of numbers together and output the result.
Simple enough, right?
Well, actually no. Even for us humans this is not so straightforward for sequences longer than, say, 7 digits.
But how do you teach a computer to do this? We will see this in a minute.
Why exactly this task? No particular reason, it was the first sequence related task that came to my
mind and I wanted to see if I could actually solve it using RNNs.


## RNN crash course

Before we start diving into how RNNs can help you solve this kind of task, a quick overview.
RNNs are, as the name states, recurrent in nature. This means, that they not only pass information along the layers
as feed-forward NNs do, but also recurrently, within a layer.

![Alt text](imgs/nn_comparison.jpg)

As can be seen in this picture, really the only difference between a plain-old NN and an RNN are the 
recurrent connections in the hidden layer. Note, that the single nodes in this picture can represent
multiple, hundreds of nodes, I just did not want to draw more lines than necessary.
So, in principle there is nothing to be afraid of. RNNs = NN + Recurrent connection.
But that does not really help (at least not me) with understanding why RNNs would be suited for sequential tasks.
A different representation sheds some light on the matter.

![Alt text](imgs/rnn_unfold.jpg)

I added some additional info to this picture, but let's focus on the main difference.
If you take the RNN from before, unpin the recurrent connections, make a copy of the whole network,
place it next to itself and then re-attach the recurrent connection to the copy, then you have
unfolded the RNN. We create exactly as many copies of the network, as we have elements in the input sequence.
Note that the recurrent connections are exactly the same between each of the copies. In more technical terms,
the weights (denoted here by R) are identical for each copy. So is the layer structure and also the
weights W, between input and hidden layer and V, between hidden and output layer.
With this unfolded RNN, we can feed in the input sequence, one element at a time, to get hidden states a and
outputs y for each element.
And that's basically it. We now have a NN that can process a sequence. And the cool thing is, we can
use all the normal tools for training this NN. We just need some loss function that measures
the difference between the NN outputs y and some targets and let it train on some data.
Of course, the internals of how to calculate gradients and update the weights are a little different,
but that need not be of concern here.

As a quick exercise, think of you would do the same task without recurrent connections.
You could for example create a network with as many input neurons as there are elements in the sequence.
But then you could not handle sequences of different length. You could condense the sequence down into a fixed-length object, but
then you would loose the sequential information altogether. So RNNs, clearly are a wildly different thing from "normal" NNs.

