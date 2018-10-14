# SNN

Stratified Neural Networks (SNNs), not to be confused with Spiking Neural Networks, is an achitecture for neural networks that is currently under development by undergraduate students Joseph Acevedo, Reeshad Arian and Archie Shahidullah, and being tested in hopes of being used to create a different network architecture that can allow for more built in complexity.

The defining aspect of an SNN are the Stratum. Each Stratum in an SNN is an entire network in itself. The network mutates in ways similar to how a Compositional Pattern-Producing Network (CPPN) does in that the activation function of each neuron can be changed to allow for 'bundling' of neurons.

Each network is fully connected, with every neuron being connected to all other neurons. Neurons with the same activation function are able to impact other neurons in their 'bundle' more heavily, allowing for linking of neurons together to simulate how the human brain creates thoughts and actions.

![alt text](http://i67.tinypic.com/2exo7sp.png)

As shown in the image each Stratum contains one or more 'bundles' of neurons. Each bundle is determined by the activation function of the neurons, and as such can be modified based on the problem and training.

For each Stratum the activation funcations of each neuron in its network is modified by the Stratum previous to it. The information is 'propagated' downwards after the initial guess. A guess is made, then the error is calculated and fed into network *N<sub>n</sub>*, where *n* is the number of Stratum that the SNN contains.

![alt text](http://i68.tinypic.com/epkpvp.png)

In the example SNN above *n* would be 2, so the calculated error after every guess would be sent to *N<sub>2</sub>* where it would then be propagated downwards, modifying every Stratum below it.

**Initial Network**
The initial testing with the SNN is done using a basic feedforward network for each Stratum. Each Stratum is trained using backpropagation, and the SNN was tested on a very simple problem.

Given a list *{a<sub>0</sub>, a<sub>1</sub>,..., a<sub>m</sub>,}*, where *m* is a positive integer that represents the length of the list you want the SNN to run on, and that list contains numbers *[1,k]*, where *k* is the maximum value integer that will be in the list, the SNN must return a list containing all numbers *1-k* ordered from appearing most often in the list, to least often.

For example:

Given the list *{1, 3, 2, 3, 4, 5, 5, 3}* the output would would be *{3, 5, 1, 2, 4}*. This is the required answer because *3* appears the most in the input list so it would appear first in the output list, followed by *5*. The numberes *1*, *2* and *4* are then ordered in the output list based on which came first in the input list.

The SNN is given an ordered list to begin with, *{1, 2,..., k}*, and the only operation it can perform on the list to get to the correct answer is to swap two elements in the list.

**More information will be posted at a later date. The results of this test should be available soon**

The next test will be using a modified version of NEAT for every Stratum, where a new innovation is introducing a different type of activation function to the Stratum. All neurons begin with the same activation function, and through innovations they can be changed.
