require "bundler/setup"
require "rann"

xor_inputs = [[0,0],[0,1],[1,0],[1,1]]
xor_targets = [[0],[1],[1],[0]]

time = Time.now.to_i
results =
  Array.new(100) do |j|
    # inputs
    inputs = Array.new(2){ |i| RANN::Neuron.new "input #{i}", 0, :input }

    # hidden layer
    hiddens = Array.new(3){ |i| RANN::Neuron.new "hidden #{i}", 3 }
    bias = RANN::Neuron.new "bias", 0, :bias

    # output layer
    output = RANN::Neuron.new "output", 3, :output, :sig

    # connect it all
    connections = []
    hiddens.each do |h|
      inputs.each do |i|
        connections.push RANN::Connection.new i, h
      end
      connections.push RANN::Connection.new bias, h
      connections.push RANN::Connection.new h, output
    end

    network = RANN::Network.new connections
    backprop = RANN::Backprop.new network

    i = 0
    loop do
      i += 1
      sample_index = (rand * xor_inputs.size).to_i

      avg_error =
        backprop.run_batch(
          [xor_inputs[sample_index].map(&:to_d)],
          [xor_targets[sample_index].map(&:to_d)],
          processes: 0,
          checking: false
        )

      break if avg_error < 0.0001
    end

    puts j
    i
  end

taken = Time.now.to_i - time
puts results.reduce(:+).fdiv(results.size).round(2)
puts "in #{taken}s"