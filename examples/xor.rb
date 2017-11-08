require "bundler/setup"
require "rann"

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

inputs = [[0,0],[0,1],[1,0],[1,1]]
targets = [[0],[1],[1],[0]]

i = 0
loop do
  i += 1
  sample_index = (rand * inputs.size).to_i

  avg_error =
    backprop.run_batch(
      [inputs[sample_index].map(&:to_d)],
      [targets[sample_index].map(&:to_d)],
      processes: 0,
      checking: false
    )

  puts "iteration #{i} error: #{avg_error.to_f}"

  break if avg_error < 0.0001
end
