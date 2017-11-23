module RANN
  class GradientChecker
    EPSILON = 10.to_d.power -4, 10

    def self.check network, inputs, targets, dvec
      gradapprox = []

      network.params.size.times do |i|
        thetaplus = network.params.dup
        thetaplus[i] = thetaplus[i] + EPSILON
        thetaminus = network.params.dup
        thetaminus[i] = thetaminus[i] - EPSILON

        network.impose thetaplus
        outputs = network.evaluate inputs
        error_thetaplus = error outputs, targets
        network.reset!

        network.impose thetaminus
        outputs = network.evaluate inputs
        error_thetaminus = error outputs, targets
        network.reset!

        gradapprox[i] = (error_thetaplus - error_thetaminus).div(EPSILON.mult(2, 10), 10)
      end

      gradapprox.each.with_index.with_object [] do |(ga, i), res|
        res << i unless in_epsilon? ga, dvec[i]
      end
    end

    def self.error outputs, targets
      total_squared_error = 0.to_d

      targets.size.times do |i|
        total_squared_error += (targets[i] - outputs[i]).power(2, 10).div(2, 10)
      end

      total_squared_error
    end

    def self.in_epsilon? exp, act, epsilon = 0.001
      # delta = [exp.abs, act.abs].min * epsilon
      delta = epsilon
      n = (exp - act).abs
      msg = "Expected |#{exp} - #{act}| (#{n}) to be <= #{delta}"

      if delta >= n
        true
      else
        puts msg

        false
      end
    end
  end
end
