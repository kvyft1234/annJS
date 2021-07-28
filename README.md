# annJS
##### made ann(artificial neural network) in javascript.
### https://kvyft1234.github.io/annJS/neuralNetwork.html

```
learn(); // xor ann learns
view(); // view xor ann
```

```
AI.ANN.setLayer([4,3,3,2]) // inputLayer(4), hiddenLayer(3,3), outputLayer(2)
```
can make ANN layer.

```
AI.ANN.variable // perceptron of value , weight, bias, wx+b, activationFunction, ect.
```
is informaion of perceptron(including weight, bias, ect).

```
AI.ANN.propagation([1,2,3,4]) // inputLayerValue
```
can propagation.

```
AI.ANN.backPropagaton([1,2],0.1) // outputLayerValue(answer) and learnRate
```
can backpropagation
