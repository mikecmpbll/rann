require "bigdecimal"
require "bigdecimal/util"

# refactor to matrix stuff blah blah
module RANN
  module Optimisers
    class RMSProp
      def initialize opts = {}, restore = {}
        @decay               = opts[:decay] || 0.9.to_d
        @fudge_factor        = opts[:fudge_factor] || 0.00000001.to_d
        @learning_rate       = opts[:learning_rate] || 0.001.to_d
        @historical_gradient = (restore[:historical_gradient] || {}).tap{ |h| h.default = 0.to_d }
      end

      def update grad, cid
        @historical_gradient[cid] = @decay.mult(@historical_gradient[cid], 10) + (1 - @decay).mult(grad.power(2, 10), 10)

        grad.mult(- @learning_rate.div(@fudge_factor + @historical_gradient[cid].sqrt(10), 10), 10)
      end
    end
  end
end
