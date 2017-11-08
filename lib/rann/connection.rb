require "securerandom"
require "bigdecimal"
require "bigdecimal/util"

module RANN
  class Connection
    attr_accessor *%i(
      output_neuron
      input_neuron
      weight
      processed
      enabled
      id
    )

    def initialize input_neuron, output_neuron, weight = nil
      @id            = SecureRandom.hex
      @output_neuron = output_neuron
      @input_neuron  = input_neuron
      @weight        = weight || initial_weight
      @processed     = false
      @enabled       = true
      @locked        = false
    end

    def process
      if processable? && !processed?
        out_value = input_neuron.value.mult weight, 10
        output_neuron.push_value! out_value
        @processed = true
      end
    end

    def neurons
      [output_neuron, input_neuron]
    end

    def processable?
      input_neuron.value
    end

    def enabled?
      enabled
    end

    def processed?
      processed
    end

    def locked?
      @locked
    end

    def reset!
      @processed = false
    end

  private
    def initial_weight
      if output_neuron.context?
        1.to_d
      else
        rand.to_d 10
      end
    end
  end
end
