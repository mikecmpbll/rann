require "rann/network"
require "rann/neuron"
require "rann/product_neuron"
require "rann/connection"
require "rann/locked_connection"

module RANN
  class LSTM
    attr_reader :network, :inputs, :outputs, :name

    def initialize name, size
      @name    = name
      @network = RANN::Network.new
      @inputs  = []
      @outputs = []
      @size    = size
      init
    end

    def init
      @size.times do |j|
        input = RANN::Neuron.new("LSTM #{name} Input #{j}", 0, :standard).tap{ |n| @network.add n }
        @inputs << input

        f = RANN::Neuron.new("LSTM #{name} F #{j}", 3, :standard, :sig).tap{ |n| @network.add n }
        i = RANN::Neuron.new("LSTM #{name} I #{j}", 4, :standard, :sig).tap{ |n| @network.add n }
        g = RANN::Neuron.new("LSTM #{name} G #{j}", 3, :standard, :tanh).tap{ |n| @network.add n }
        o = RANN::Neuron.new("LSTM #{name} O #{j}", 3, :standard, :sig).tap{ |n| @network.add n }
        bias_f = RANN::Neuron.new("LSTM #{name} Bias F #{j}", 0, :bias).tap{ |n| @network.add n }
        bias_i = RANN::Neuron.new("LSTM #{name} Bias I #{j}", 0, :bias).tap{ |n| @network.add n }
        bias_g = RANN::Neuron.new("LSTM #{name} Bias G #{j}", 0, :bias).tap{ |n| @network.add n }
        bias_o = RANN::Neuron.new("LSTM #{name} Bias O #{j}", 0, :bias).tap{ |n| @network.add n }
        memory_product = RANN::ProductNeuron.new("LSTM #{name} Mem Product #{j}", 2, :standard, :linear).tap{ |n| @network.add n }
        i_g_product = RANN::ProductNeuron.new("LSTM #{name} Hidden 2/3 Product #{j}", 2, :standard, :linear).tap{ |n| @network.add n }
        memory_standard = RANN::Neuron.new("LSTM #{name} Mem Standard #{j}", 2, :standard, :linear).tap{ |n| @network.add n }
        memory_tanh = RANN::Neuron.new("LSTM #{name} Mem Tanh #{j}", 1, :standard, :tanh).tap{ |n| @network.add n }
        memory_o_product = RANN::ProductNeuron.new("LSTM #{name} Mem/Hidden 4 Product #{j}", 2, :standard, :linear).tap{ |n| @network.add n }
        output = RANN::Neuron.new("LSTM #{name} Output #{j}", 1, :standard, :linear).tap{ |n| @network.add n }
        @outputs << output
        memory_context = RANN::Neuron.new("LSTM #{name} Mem Context #{j}", 1, :context).tap{ |n| @network.add n }
        output_context = RANN::Neuron.new("LSTM #{name} Output Context #{j}", 1, :context).tap{ |n| @network.add n }

        @network.add RANN::Connection.new input, f
        @network.add RANN::Connection.new input, i
        @network.add RANN::Connection.new input, g
        @network.add RANN::Connection.new input, o
        @network.add RANN::LockedConnection.new f, memory_product, 1.to_d
        @network.add RANN::LockedConnection.new i, i_g_product, 1.to_d
        @network.add RANN::LockedConnection.new g, i_g_product, 1.to_d
        @network.add RANN::LockedConnection.new i_g_product, memory_standard, 1.to_d
        @network.add RANN::LockedConnection.new memory_product, memory_standard, 1.to_d
        @network.add RANN::LockedConnection.new memory_standard, memory_tanh, 1.to_d
        @network.add RANN::LockedConnection.new o, memory_o_product, 1.to_d
        @network.add RANN::LockedConnection.new memory_tanh, memory_o_product, 1.to_d
        @network.add RANN::LockedConnection.new memory_o_product, output, 1.to_d
        @network.add RANN::LockedConnection.new memory_standard, memory_context, 1.to_d
        @network.add RANN::LockedConnection.new memory_context, memory_product, 1.to_d
        @network.add RANN::LockedConnection.new memory_context, i, 1.to_d
        @network.add RANN::LockedConnection.new memory_o_product, output_context, 1.to_d
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
      @inputs.each do |input|
        @network.add RANN::Connection.new neuron, input
      end
    end
  end
end
