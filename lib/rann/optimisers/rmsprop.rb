require "bigdecimal"
require "bigdecimal/util"

# refactor to matrix stuff blah blah
module RANN
  module Optimisers
    class RMSProp
      def initialize opts = {}
        @decay               = opts[:decay] || 0.9.to_d
        @fudge_factor        = opts[:fudge_factor] || 0.00000001.to_d
        @learning_rate       = opts[:learning_rate] || 0.01.to_d
        @historical_gradient = {}.tap{ |h| h.default = 0.to_d }
      end

      def update grad, cid
        @historical_gradient[cid] = @decay.mult(@historical_gradient[cid], RANN.d) + (1 - @decay).mult(grad.power(2, RANN.d), RANN.d)

        grad.mult(- @learning_rate.div(@fudge_factor + @historical_gradient[cid].sqrt(RANN.d), RANN.d), RANN.d)
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
