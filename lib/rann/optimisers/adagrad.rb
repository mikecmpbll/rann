require "bigdecimal"
require "bigdecimal/util"

# refactor to matrix stuff blah blah
module RANN
  module Optimisers
    class AdaGrad
      def initialize opts = {}
        @fudge_factor        = opts[:fudge_factor] || 0.00000001.to_d
        @learning_rate       = opts[:learning_rate] || 0.1.to_d
        @historical_gradient = {}.tap{ |h| h.default = 0.to_d }
      end

      def update grad, cid
        @historical_gradient[cid] = @historical_gradient[cid] + grad ** 2

        grad * - @learning_rate / (@fudge_factor + @historical_gradient[cid].sqrt(0))
      end

      # anything that gets modified over the course of training
      def state
        {
          historical_gradient: @historical_gradient,
        }
      end

      def load_state state
        state.each do |name, value|
          instance_variable_set("@#{name}", value)
        end
      end
    end
  end
end
