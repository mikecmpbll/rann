require "bigdecimal"
require "bigdecimal/util"
require "rann/version"
require "rann/network"
require "rann/neuron"
require "rann/product_neuron"
require "rann/connection"
require "rann/locked_connection"
require "rann/backprop"

module RANN
  @@significant_digits = 10

  def self.significant_digits= sd
    @@significant_digits = sd
  end

  def self.d
    @@significant_digits
  end
end
