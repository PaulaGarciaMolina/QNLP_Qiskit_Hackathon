# QNLP Qiskit Hackathon
Quantum Natural Language Processing (QNLP)'s project repository for Qiskit Hackathon Europe. Team members: Edwin Agnew, Pablo Díez-Valle, Paula García-Molina, María Hita-Pérez,  and Carlos Vega

## Outline
This repository is split into two directories:
- *semantic_interpretation*: notebooks and datasets for determining similarity between English and Spanish sentences
- *sentiment_analysis*: notebooks and datasets for classifying sentences into one of your sentiments: happy, sad, angry, scared 

## Philosophy
QNLP treats language as a quantum process and interpets sentences as circuits by using categorical quantum mechanics and the ZX-calculus. Rather than assigning meaning to individual words, the key focus is on *how meaning composes*. 

For an excellent introduction to the mathematical foundations of this approach, see ([Coecke et al.](https://arxiv.org/abs/2012.03755)).

## Procedure
Our methodology is largely based off ([Lorenz et al.](https://arxiv.org/abs/2102.12846)) and uses the following structure:
- Create a dataset of sentences and split into training and testing sets
- Convert a sentence into a DisCoPy diagram:
  - Define n (noun) and s (sentence) as basic types 
  - Derive types for other word: n @ n.l for English adjectives, n.r @ s @ n.l for transitive verbs, etc.
  - Parse a sentence according to its grammatical structure and plug the words together using cups and caps according to their respective types
- Convert sentence diagrams into parameterized quantum circuits
  - Choose how many qubits are required to represent the basic types: 2 qubits for sentences and 1 qubit for nouns, 2 and 2 or 1 and 1.
  - Choose an ansatz (we used the IQP ansatz due to its effectiveness in ([Lorenz et al.](https://arxiv.org/abs/2102.12846)))
  - Construct a functor which turns a sentence diagram into a parameterized quantum circuit, based off the chosen ansatz. 
- Optimize parameters by applying the SPSA variational algorithm on training sentences and cross entropy as the cost function (or train error). This differed slightly in each of our implementations:
  - **sentiment analysis**:
    - *2 qubit multi-classification* : train a single classifier evaluated on the entire training set. The 2 qubit output (|00>, |01>, |10> or |11>) corresponds to one of our four sentiments. 
    - *1 versus 1*: Train 6 1v1 binary classifiers (happy v sad, happy v angry, etc.), each evaluated on a restricted training set of sentences corresponding to either label. The most popular prediction is chosen as the final output. By using a binary classifier, a sentence can be represented with a single qubit which halves the number of parameters to be optimized.
    - *1 versus all*: Train 4 binary classifiers (happy v not happy, sad v not sad, etc.), each evaluated on the entire training set with modified labels. The largest amplitude is chosen as the final prediction. Note that in our results, this was trained with a more limited dataset.
  - **semantic interpretation**:
    - Train a single classifier which computes the similarity between the 2 output qubits of an English sentence circuit and the 2 output qubits of a Spanish circuit. We have tested datasets of different sizes:
      - *Small*: 20 English sentences and 20 Spanish sentences.
      - *Medium*: 65 English sentences and 65 Spanish sentences.
      - *Big*: 90 English sentences and 90 Spanish sentences.   

## Results
We obtained the following results:
### Semantic interpretation: 
The test set accuracies are (the cost function appears between brackets):
#### Statevector simulator:
- *Small*: 20% (train error).
- *Medium*: 65% (train error).
- *Big*:  62% (train error).
#### Qasm simulator:
- *Small*: 75% (train error).
- *Medium*: 62% (train error).
- *Big*:  53% (train error).
#### ibmq_16_melbourne noisy simulator:
- *Small*: 64% (train error).
- *Medium*: 72% (train error).
- *Big*:  61% (train error).

We performed 1000 iterations of the SPSA optimizer for the Statevector simulator (all sizes), 2000 iterations for the QASM simulator (all sizes), and 2000, 1309, 928 iterations for the noisy simulations with small, medium, and big datasets respectively.
### Sentiment analysis: 
#### Statevector simulator:
- 1vs1: 63% (cross entropy), 77% (train error).
- 1vsAll: 75% (cross entropy).
- 2-qubit: 81 % (cross entropy).
#### Qasm simulator:
- 1vs1: 79% (cross entropy), 87% (train error).
#### ibmq_16_melbourne noisy simulator:
- 1vs1: 78% (cross entropy).

1vs1 results are for 2000 iterations of the SPSA optimizer, except for the noisy simulations (500 iterations) due to time requirements. For 1vsAll and 2-qubit multi-class classification we perform 1250 iterations.

## Notes

- For more details see our report here (link).
- Our results could easily be expanded to include more qasm simulations and real device experiments, but we were unable to report them here because of time constraints and the code being quite slow since it has to create and evaluate 2 circuits per train sentence per iteration.
- Although not used in our project, a DiscoCat sentence parser is now available at https://qnlp.cambridgequantum.com/generate.html


Packages used:
- qiskit 0.26.2
- discopy 0.3.5
- pytket 0.7.2
