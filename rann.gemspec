
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "rann/version"

Gem::Specification.new do |spec|
  spec.name          = "rann"
  spec.version       = RANN::VERSION
  spec.authors       = ["Michael Campbell"]
  spec.email         = ["mike@ydd.io"]

  spec.summary       = %q{Ruby Artificial Neural Networks}
  spec.description   = %q{Libary for working with neural networks in Ruby.}
  spec.homepage      = "https://github.com/mikecmpbll/rann"

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}){ |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_runtime_dependency "parallel", "~> 1.12.0"
  spec.add_runtime_dependency "ruby-graphviz", "~> 1.2.3"

  spec.add_development_dependency "bundler", "~> 1.16"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "minitest", "~> 5.0"
end
