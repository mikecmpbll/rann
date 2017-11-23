require "bigdecimal"
require "bigdecimal/util"
require "parallel"
require "rann/gradient_checker"
require "rann/util/array_ext"
require "rann/optimisers/adagrad"
require "rann/optimisers/rmsprop"

module RANN
  class Backprop
    include Util::ArrayExt

    ACTIVATION_DERIVATIVES = {
      relu:   ->(x){ x > 0 ? 1.to_d : 0.to_d },
      sig:    ->(x){ x.mult(1 - x, 10) },
      linear: ->(_){ 1.to_d },
      tanh:   ->(x){ 1 - x.power(2, 10) },
      step:   ->(_){ 0.to_d },
    }

    attr_accessor :network

    def initialize network, opts = {}, restore = {}
      @network          = network
      @connections_hash = network.connections.each.with_object({}){ |c, h| h[c.id] = c }
      @optimiser        = RANN::Optimisers.const_get(opts[:optimiser] || 'RMSProp').new opts, restore
      @batch_count      = 0.to_d
    end

    def run_batch inputs, targets, opts = {}
      @batch_count += 1

      batch_size      = inputs.size
      avg_gradients   = Hash.new{ |h, k| h[k] = 0 }
      avg_batch_error = 0

      # force longer bits of work per iteration, to maximise CPU usage
      # less marshalling data etc, more work.
      grouped_inputs  = in_groups(inputs, [1, opts[:processes]].max * 10, false).reject &:empty?
      reduce_proc =
        lambda do |_, _, result|
          group_avg_gradients, group_avg_error = result

          avg_gradients.merge!(group_avg_gradients){ |_, o, n| o + n }
          avg_batch_error += group_avg_error
        end

      Parallel.each_with_index(
        grouped_inputs,
        in_processes: opts[:processes],
        finish: reduce_proc
      ) do |inputs, i|
        group_avg_gradients = Hash.new{ |h, k| h[k] = 0.to_d }
        group_avg_error = 0.to_d

        inputs.each_with_index do |input, j|
          gradients, error = Backprop.run_single network, input, targets[i + j]

          gradients.each do |cid, g|
            group_avg_gradients[cid] += g.div batch_size, 10
          end
          group_avg_error += error.div batch_size, 10
        end

        group_avg_gradients.default_proc = nil
        [group_avg_gradients, group_avg_error]
      end

      if opts[:checking]
        # check assumes batchsize 1 for now
        sorted_gradients = avg_gradients.values_at *network.connections.map(&:id)
        invalid = GradientChecker.check network, inputs.first, targets.first, sorted_gradients
        if invalid.empty?
          puts "gradient valid"
        else
          puts "gradients INVALID for connections:"
          invalid.each do |i|
            puts "#{network.connections[i].input_neuron.name} -> #{network.connections[i].output_neuron.name}"
          end
        end
      end

      avg_gradients.each do |con_id, gradient|
        con = @connections_hash[con_id]
        next if con.locked?

        update = @optimiser.update gradient, con.id

        con.weight += update
      end

      avg_batch_error
    end

    def self.run_single network, inputs, targets
      states = []
      inputs = [inputs] if inputs.flatten == inputs

      # run the data into the network. (feed forward)
      # all but last
      (inputs.size - 1).times do |timestep|
        network.evaluate inputs[timestep]
        states[timestep] = network.reset!
      end
      # last
      outputs = network.evaluate inputs.last
      states[inputs.size - 1] = network.reset!

      # calculate error
      error = mse targets, outputs

      # backward pass with unravelling for recurrent networks
      node_deltas = Hash.new{ |h, k| h[k] = Hash.new(0.to_d) }
      gradients = Hash.new(0)

      initial_timestep = inputs.size - 1
      neuron_stack = network.output_neurons.map{ |n| [n, initial_timestep] }

      while current = neuron_stack.shift
        neuron, timestep = current
        next if node_deltas[timestep].key? neuron

        from_here = bptt_connecting_to neuron, network, timestep
        neuron_stack.push *from_here

        # neuron delta is summation of neuron deltas deltas for the connections
        # from this neuron
        node_delta =
          if neuron.output?
            output_index = network.output_neurons.index neuron
            activation_derivative = ACTIVATION_DERIVATIVES[neuron.activation_function]
            mse_delta targets[output_index], outputs[output_index], activation_derivative
          else
            sum_of_deltas =
              network.connections_from(neuron).reduce 0.to_d do |m, c|
                out_timestep = c.output_neuron.context? ? timestep + 1 : timestep
                output_node_delta = node_deltas[out_timestep][c.output_neuron.id]

                # connection delta is the output neuron delta multiplied by the
                # connection's weight
                connection_delta =
                  if c.output_neuron.is_a? ProductNeuron
                    intermediate = states[out_timestep][:intermediates][c.output_neuron.id]
                    output_node_delta.mult intermediate.div(states[timestep][:values][c.input_neuron.id], 10), 10
                  else
                    output_node_delta.mult c.weight, 10
                  end

                m + connection_delta
              end

            ACTIVATION_DERIVATIVES[neuron.activation_function]
              .call(states[timestep][:values][neuron.id])
              .mult(sum_of_deltas, 10)
          end

        node_deltas[timestep][neuron.id] = node_delta

        network.connections_to(neuron).each do |c|
          in_timestep = neuron.context? ? timestep - 1 : timestep

          # connection gradient is the output neuron delta multipled by the
          # connection's input neuron value.
          gradient =
            if c.output_neuron.is_a? ProductNeuron
              intermediate = states[timestep][:intermediates][c.output_neuron.id]
              node_delta.mult intermediate.div(c.weight, 10), 10
            elsif c.input_neuron.context? && timestep == 0
              0.to_d
            else
              node_delta.mult states[in_timestep][:values][c.input_neuron.id], 10
            end

          gradients[c.id] += gradient
        end
      end

      reset! network
      [gradients, error]
    end

    def state
      { historical_gradient: @historical_gradient }
    end

    def self.reset! network
      network.reset!
      network.neurons.select(&:context?).each{ |n| n.value = 0.to_d }
    end

    def adagrad avg_grad, cid
      @historical_gradient[cid] = DECAY.mult(@historical_gradient[cid], 10) + (1 - DECAY).mult(avg_grad.power(2, 10), 10)

      avg_grad.mult(- @lr.div((FUDGE_FACTOR + @historical_gradient[cid]).sqrt(10), 10), 10)
    end

    def self.mse targets, outputs
      total_squared_error = 0.to_d

      targets.size.times do |i|
        total_squared_error += (targets[i] - outputs[i]).power(2, 10).div(2, 10)
      end

      total_squared_error
    end

    def self.mse_delta target, actual, activation_derivative
      step_one = actual - target
      step_two = activation_derivative.call actual

      step_one.mult step_two, 10
    end

    def self.bptt_connecting_to neuron, network, timestep
      # halt traversal if we're at a context and we're at the base timestep
      return [] if neuron.context? && timestep == 0

      network.connections_to(neuron).each.with_object [] do |c, a|
        # don't enqueue connections from inputs
        next if c.input_neuron.input?

        timestep -= timestep if neuron.context?
        a << [c.input_neuron, timestep]
      end
    end
  end
end
