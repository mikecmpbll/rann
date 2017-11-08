# RANN

This library provides objects and algorithms for designing, processing and
training Artificial Neural Networks in Ruby.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'rann'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install rann

## Usage

See examples/

To run an example:

```
git clone https://github.com/mikecmpbll/rann.git
cd rann
bin/setup
ruby examples/xor.rb
```

## TODO

So much. So much.

- Convenience methods for setting up standard network topologies, crucially,
  layers
- Batch normalization/drop out/early stopping
- Hyperparameter optimisation
- Other adaptive learning rate algorithms (Adadelta, Adam, etc?)
- Explore matrix operations and other ways to optimise performance of algorithms
- RPROP?
- Use enumerable-statistics gem?
- Speed up by adding a reduce step to the parallel gem?
- More examples
- Tests

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/mikecmpbll/rann.
