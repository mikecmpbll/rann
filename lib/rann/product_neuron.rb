require "rann/neuron"

module RANN
  class ProductNeuron < Neuron
    def set_value!
      intermediate = incoming.reduce{ |i, m| m.mult i, 10 }
      self.value   = ACTIVATION_FUNCTIONS[activation_function].call intermediate
    end
  end
end
