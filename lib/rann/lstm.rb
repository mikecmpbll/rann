require "rann/network"
require "rann/neuron"
require "rann/product_neuron"
require "rann/connection"
require "rann/locked_connection"

module RANN
  class LSTM
    attr_reader :network, :inputs, :outputs, :name

    def initialize name
      @name    = name
      @network = RANN::Network.new
      @inputs  = []
      @outputs = []
    end

    def init
      @inputs.each.with_index do |input, i|
        f = RANN::Neuron.new("LSTM #{name} F #{i}", 3, :standard, :sig).tap{ |n| @network.add n }
        i = RANN::Neuron.new("LSTM #{name} I #{i}", 4, :standard, :sig).tap{ |n| @network.add n }
        g = RANN::Neuron.new("LSTM #{name} G #{i}", 3, :standard, :tanh).tap{ |n| @network.add n }
        o = RANN::Neuron.new("LSTM #{name} O #{i}", 3, :standard, :sig).tap{ |n| @network.add n }
        bias_f = RANN::Neuron.new("LSTM #{name} Bias F #{i}", 0, :bias).tap do |n|
          @network.add n
          n.value = 1.to_d
        end
        bias_i = RANN::Neuron.new("LSTM #{name} Bias I #{i}", 0, :bias).tap do |n|
          @network.add n
          n.value = 1.to_d
        end
        bias_g = RANN::Neuron.new("LSTM #{name} Bias G #{i}", 0, :bias).tap do |n|
          @network.add n
          n.value = 1.to_d
        end
        bias_o = RANN::Neuron.new("LSTM #{name} Bias O #{i}", 0, :bias).tap do |n|
          @network.add n
          n.value = 1.to_d
        end
        memory_product = RANN::ProductNeuron.new("LSTM #{name} Mem Product #{i}", 2, :standard, :linear).tap{ |n| @network.add n }
        i_g_product = RANN::ProductNeuron.new("LSTM #{name} Hidden 2/3 Product #{i}", 2, :standard, :linear).tap{ |n| @network.add n }
        memory_standard = RANN::Neuron.new("LSTM #{name} Mem Standard #{i}", 2, :standard, :linear).tap{ |n| @network.add n }
        memory_tanh = RANN::Neuron.new("LSTM #{name} Mem Tanh #{i}", 1, :standard, :tanh).tap{ |n| @network.add n }
        memory_o_product = RANN::ProductNeuron.new("LSTM #{name} Mem/Hidden 4 Product #{i}", 2, :standard, :linear).tap{ |n| @network.add n }
        output = RANN::Neuron.new("LSTM #{name} Output #{i}", 1, :standard, :linear).tap{ |n| @network.add n }
        @outputs << output
        memory_context = RANN::Neuron.new("LSTM #{name} Mem Context #{i}", 1, :context).tap{ |n| @network.add n }
        output_context = RANN::Neuron.new("LSTM #{name} Output Context #{i}", 1, :context).tap{ |n| @network.add n }

        @network.add RANN::LockedConnection.new input, f, 1
        @network.add RANN::LockedConnection.new input, i, 1
        @network.add RANN::LockedConnection.new input, g, 1
        @network.add RANN::LockedConnection.new input, o, 1
        @network.add RANN::LockedConnection.new f, memory_product, 1
        @network.add RANN::LockedConnection.new i, i_g_product, 1
        @network.add RANN::LockedConnection.new g, i_g_product, 1
        @network.add RANN::LockedConnection.new i_g_product, memory_standard, 1
        @network.add RANN::LockedConnection.new memory_product, memory_standard, 1
        @network.add RANN::LockedConnection.new memory_standard, memory_tanh, 1
        @network.add RANN::LockedConnection.new o, memory_o_product, 1
        @network.add RANN::LockedConnection.new memory_tanh, memory_o_product, 1
        @network.add RANN::LockedConnection.new memory_o_product, output, 1
        @network.add RANN::LockedConnection.new memory_standard, memory_context, 1
        @network.add RANN::Connection.new memory_context, memory_product
        @network.add RANN::Connection.new memory_context, i
        @network.add RANN::LockedConnection.new memory_o_product, output_context, 1
        @network.add RANN::Connection.new output_context, f
        @network.add RANN::Connection.new output_context, i
        @network.add RANN::Connection.new output_context, g
        @network.add RANN::Connection.new output_context, o
        @network.add RANN::Connection.new bias_f, f
        @network.add RANN::Connection.new bias_i, i
        @network.add RANN::Connection.new bias_g, g
        @network.add RANN::Connection.new bias_o, o
      end
    end

    def add_input neuron
      input = RANN::Neuron.new "LSTM #{name} Input #{neuron.name}", 0, :standard, :linear
      @network.add input
      @inputs << input
      connection = RANN::Connection.new neuron, input
      @network.add connection
    end
  end
end
