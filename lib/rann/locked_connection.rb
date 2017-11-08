require "rann/connection"

module RANN
  class LockedConnection < Connection
    def initialize(*)
      super
      @locked = true
    end
  end
end
