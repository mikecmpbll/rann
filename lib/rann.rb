require "bigdecimal"
require "bigdecimal/util"
require "rann/version"
require "rann/network"
require "rann/neuron"
require "rann/product_neuron"
require "rann/connection"
require "rann/locked_connection"
require "rann/backprop"

BigDecimal.mode BigDecimal::EXCEPTION_ALL, true
BigDecimal.limit 10
