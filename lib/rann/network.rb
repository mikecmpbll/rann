require "graphviz"
require "yaml"

module RANN
  class Network
    UnconnectedNetworkError = Class.new(StandardError)

    attr_accessor *%i(
      neurons
      input_neurons
      hidden_neurons
      output_neurons
      connections
      structure
    )

    def initialize connections = []
      @connections    = connections
      @neurons        = connections.flat_map(&:neurons).uniq
      @input_neurons  = @neurons.select &:input?
      @output_neurons = @neurons.select &:output?
      @hidden_neurons = @neurons - @input_neurons - @output_neurons
    end

    def impose weights
      connections.each.with_index do |c, i|
        c.weight = weights[i]
      end
    end

    def params
      connections.map(&:weight)
    end

    def evaluate input
      input_neurons.each.with_index do |neuron, i|
        neuron.value = input[i]
      end

      # use some proper graph traversal, rather than this crude blanketing?
      # would probably be easier to detect circular dependency this way too?
      begin
        i = 0
        until output_neurons.all?{ |neuron| neuron.value }
          i += 1
          connections.each do |connection|
            next if !connection.enabled?

            connection.process
          end
          raise UnconnectedNetworkError if i > 5_000
        end
      rescue UnconnectedNetworkError
        visualise
        raise
      end

      outputs
    end

    def visualise
      # Create a new graph
      g = GraphViz.new(:G, type: :digraph)

      # Create nodes

      missing_nodes = connections.each.with_object([]) do |c, o|
        o << c.output_neuron unless neurons.include? c.output_neuron
        o << c.input_neuron unless neurons.include? c.input_neuron
      end

      graph_nodes = neurons.each.with_object({}) do |n, h|
        h[n] = g.add_nodes("#{n.name}: #{n.value&.to_f&.round(5)}")
      end

      # Create edges between the nodes
      connections.each do |c|
        g.add_edges(
          graph_nodes[c.input_neuron],
          graph_nodes[c.output_neuron],
          color: c.processed? ? "#ff0000" : "#000000",
          label: c.weight.to_f.round(5)
        )
      end

      # Generate output image
      g.output png: "nnet.png"
      `open nnet.png`
    end

    def dump_weights
      File.write "nn_weights_dump_#{DateTime.now.strftime('%Y-%m-%d-%H-%M-%S')}.yml", params.to_yaml
    end

    def outputs
      output_neurons.map &:value
    end

    def state
      neurons.each.with_object({}){ |n, s| s[n.id] = n.value }
    end

    def connections_to neuron
      @connections_to = {} unless defined? @connections_to

      @connections_to[neuron] ||= connections.select{ |con| con.output_neuron == neuron }
    end

    def connections_from neuron
      @connections_from = {} unless defined? @connections_from

      @connections_from[neuron] ||= connections.select{ |con| con.input_neuron == neuron }
    end

    def add *features
      features.each do |feature|
        case feature
        when Neuron
          case feature.type
          when :input
            @input_neurons  << feature
          when :output
            @output_neurons << feature
          else
            @hidden_neurons << feature
          end

          @neurons << feature
        when Connection
          @connections << feature
        when Network
          add *feature.neurons
          add *feature.connections
        end
      end
    end

    def remove *features
      features.each do |feature|
        case feature
        when Neuron
          case feature.type
          when :input
            raise "trying to remove an input neuron ..."
          when :output
            raise "trying to remove an output neuron ..."
          else
            @hidden_neurons.delete feature
          end

          @neurons.delete feature
        when Connection
          @connections.delete feature
        end
      end
    end

    def reset!
      state.tap do
        neurons.each{ |neuron| neuron.reset! }
        connections.each{ |connection| connection.reset! }
      end
    end

    def recalculate_neuron_connection_counts!
      neurons.each do |neuron|
        neuron.connection_count = connections.count{ |c| c.output_neuron == neuron }
      end
    end
  end
end
