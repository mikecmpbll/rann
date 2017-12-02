require "rann/neuron"

module RANN
  class ProductNeuron < Neuron
    attr_accessor :intermediate

    def set_value!
      @intermediate = incoming.reduce{ |i, m| m.mult(i, RANN.d) }
      self.value    = ACTIVATION_FUNCTIONS[activation_function].call @intermediate
    end
  end
end
