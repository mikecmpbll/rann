- So. Many. Bugs. Turns out the product neuron stuff was still broken and
  network evaluation wasn't behaving correctly with context neurons (recurrent
  connections). Also an error in timestep handling during backprop, and just
  generally network traversal ...

  *Michael Campbell*

- Don't lock the input connections into the LSTM layer, that acts as the fully
  connected part of the network and that's where the majority of learning takes
  place, derp.

  *Michael Campbell*

- Don't change the weight of a LockedConnection during initialisation. (eugh)

  *Michael Campbell*

- Fix bug in backprop over product neuron.

  *Michael Campbell*

- Worked out how to use BigDecimal properly, lol. Refactored to use
  `BigDecimal.limit`.

  *Michael Campbell*

- Allow customisable precision, defaults to 10.

  *Michael Campbell*

## 0.2.7 (December 2, 2017)

- Allow different weights to the different gates in the LSTM. Previously it was
  using one weighted connection into the LSTM and unweighted connections to the
  gates, this has been reversed.

  *Michael Campbell*

- Add save and restore methods to backprop.

  *Michael Campbell*

- Fix alias error.

  *Michael Campbell*

- Simple weight initialisation method.

  *Michael Campbell*

- Fix ProductNeuron backprop bug, and index bug in parallel code.

  *Michael Campbell*

- Total backprop refactor. Removed recursion in network traversal in favour of
  custom stack iteration and the whole thing is tons more memory efficient.

  *Michael Campbell*

- Basic classes required for most neural network designs and backprop.

  *Michael Campbell*