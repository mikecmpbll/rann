# A reproduction of the wonderful step by step backprop example at
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

require "bundler/setup"
require "rann"

# inputs
inputs = Array.new(2){ |i| RANN::Neuron.new "input #{i}", 0, :input, :sig }

# hidden layer
hiddens = Array.new(2){ |i| RANN::Neuron.new "hidden #{i}", 3, :standard, :sig }
hidden_bias = RANN::Neuron.new "bias", 0, :bias

# output layer
outputs = Array.new(2){ |i| RANN::Neuron.new "output #{i}", 3, :output, :sig }
output_bias = RANN::Neuron.new "bias", 0, :bias

# connect it all w/initial weights
connections = []
connections << RANN::Connection.new(inputs[0], hiddens[0], 0.15.to_d)
connections << RANN::Connection.new(inputs[0], hiddens[1], 0.25.to_d)
connections << RANN::Connection.new(inputs[1], hiddens[0], 0.2.to_d)
connections << RANN::Connection.new(inputs[1], hiddens[1], 0.3.to_d)
connections << RANN::Connection.new(hidden_bias, hiddens[0], 0.35.to_d)
connections << RANN::Connection.new(hidden_bias, hiddens[1], 0.35.to_d)

connections << RANN::Connection.new(hiddens[0], outputs[0], 0.4.to_d)
connections << RANN::Connection.new(hiddens[0], outputs[1], 0.5.to_d)
connections << RANN::Connection.new(hiddens[1], outputs[0], 0.45.to_d)
connections << RANN::Connection.new(hiddens[1], outputs[1], 0.55.to_d)
connections << RANN::Connection.new(output_bias, outputs[0], 0.6.to_d)
connections << RANN::Connection.new(output_bias, outputs[1], 0.6.to_d)

network = RANN::Network.new connections
backprop = RANN::Backprop.new network

inputs = [0.05.to_d, 0.10.to_d]
targets = [0.1.to_d, 0.99.to_d]
outputs = network.evaluate [0.05.to_d, 0.10.to_d]

puts "forward prop outputs: #{outputs.map(&:to_f).inspect}"

network.reset!

gradients, error =
  RANN::Backprop.run_single(
    network,
    [0.05.to_d, 0.10.to_d],
    [0.01.to_d, 0.99.to_d]
  )

puts "error: #{error.to_f}"
puts "backprop gradients & updates:"
gradients.each do |cid, g|
  c = network.connections.find{ |c| c.id == cid }
  puts "#{c.input_neuron.name} -> #{c.output_neuron.name}: g = #{g.to_f}, u = #{(c.weight - (0.5.to_d.mult(g, 10))).to_f}"
end
