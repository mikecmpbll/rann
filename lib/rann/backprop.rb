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
        if GradientChecker.check network, inputs.first, targets.first, sorted_gradients
          puts "gradient valid"
        else
          puts "gradient INVALID"
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
      deltas = Hash.new{ |h, k| h[k] = Hash.new(0.to_d) }

      # outputs first
      network.output_neurons.each.with_index do |o, i|
        activation_derivative = ACTIVATION_DERIVATIVES[o.activation_function]

        deltas[0][o.id] = mse_delta(targets[i], outputs[i], activation_derivative)
      end

      # remove this push mechanism, shouldn't be necessary and uses extra memory.
      incoming_deltas = Hash.new{ |h, k| h[k] = Hash.new{ |h, k| h[k] = [] } }

      intial_timestep = inputs.size - 1
      connection_stack =
        network.output_neurons
          .flat_map{ |n| network.connections_to n }
          .map{ |c| [c, intial_timestep] }

      # maybe change this to traverse the static network timestep times if this
      # proves too difficult to rationalise
      while current = connection_stack.shift
        conn, timestep = current

        inp_n = conn.input_neuron
        out_n = conn.output_neuron
        out_timestep = out_n.context? ? timestep + 1 : timestep

        # skip if already processed (might've been enqueued by two nodes before
        # being processed). could alternatively add a check when enqueueing that
        # not already enqueued? might be better for memory, but slow down
        # processing.
        next if deltas[timestep].key?(inp_n.id)

        from_here = bptt_connecting_to inp_n, network, timestep, deltas
        connection_stack.unshift *from_here

        incoming_deltas[timestep][inp_n.id] <<
          if out_n.is_a? ProductNeuron
            intermediate = states[out_timestep][:intermediates][out_n.id]
            deltas[out_timestep][out_n.id].mult intermediate.div(states[timestep][:values][inp_n.id], 10), 10
          else
            deltas[out_timestep][out_n.id].mult conn.weight, 10
          end

        if incoming_deltas[timestep][inp_n.id].size == network.connections_from(inp_n).size
          sum_of_deltas = incoming_deltas[timestep][inp_n.id].reduce :+

          deltas[timestep][inp_n.id] =
            ACTIVATION_DERIVATIVES[inp_n.activation_function]
              .call(states[timestep][:values][inp_n.id])
              .mult(sum_of_deltas, 10)
        end
      end

      gradients = {}

      network.connections.each_with_index do |con, i|
        gradients[con.id] = 0.to_d

        (inputs.size - 1).downto 0 do |t|
          if nd = deltas[t][con.output_neuron.id]
            gradient =
              if con.input_neuron.context?
                t == 0 ? 0.to_d : nd.mult(states[t - 1][:values][con.input_neuron.id], 10)
              elsif con.output_neuron.is_a? ProductNeuron
                intermediate = states[t][:intermediates][con.output_neuron.id]
                nd.mult intermediate.div(con.weight, 10), 10
              else
                nd.mult states[t][:values][con.input_neuron.id], 10
              end

            gradients[con.id] += gradient
          end
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

    def self.bptt_connecting_to neuron, network, timestep, deltas
      # halt traversal if we're at a context and we're at the base timestep
      return [] if neuron.context? && timestep == 0

      network.connections_to(neuron).each.with_object [] do |c, a|
        # don't enqueue connections from inputs
        next if c.input_neuron.input?

        timestep -= timestep if neuron.context?

        unless deltas[timestep].key?(c.input_neuron.id)
          a << [c, timestep]
        end
      end
    end
  end
end
