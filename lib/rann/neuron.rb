require "securerandom"
require "bigdecimal"
require "bigdecimal/util"

module RANN
  class Neuron
    ACTIVATION_FUNCTIONS = {
      sig:    ->(v){ 1.to_d.div(1 + Math::E.to_d.power(-v, 10), 10) },
      tanh:   ->(v){ Math.tanh(v).to_d(10) },
      relu:   ->(v){ [0.to_d, v].max },
      linear: ->(v){ v },
      step:   ->(v){ v > 0.5 ? 1.to_d : 0.to_d },
    }

    attr_accessor *%i(
      activation_function
      value
      incoming
      connection_count
      type
      name
      id
    )

    def initialize name, connection_count, type = :standard, af = nil
      @id                  = SecureRandom.hex
      @connection_count    = connection_count
      @type                = type
      @incoming            = []
      @activation_function = af || initial_activation_function
      @name                = name

      set_default_value!
    end

    def push_value! value
      incoming << value
      set_value! if incoming.size == connection_count
    end

    def set_value!
      intermediate = incoming.reduce :+
      self.value   = ACTIVATION_FUNCTIONS[activation_function].call intermediate
    end

    def reset!
      set_default_value!
      @incoming.clear
    end

    def increment_connection_count!
      @connection_count += 1
    end

    def decrement_connection_count!
      @connection_count -= 1
    end

    %i(input output context bias standard).each do |t|
      define_method "#{t}?" do
        type == t
      end
    end

  private
    def set_default_value!
      self.value =
        if context?
          value || 0.to_d
        elsif bias?
          1.to_d
        end
    end

    def initial_activation_function
      if standard? || context?
        :relu
      else
        :linear
      end
    end
  end
end
