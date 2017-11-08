module RANN
  module Util
    module ArrayExt
      # Method `in_groups` from
      # activesupport/lib/active_support/core_ext/array/grouping.rb under MIT
      # licence. Original licence printed below.
      #
      # Copyright (c) 2005-2017 David Heinemeier Hansson

      # Permission is hereby granted, free of charge, to any person obtaining
      # a copy of this software and associated documentation files (the
      # "Software"), to deal in the Software without restriction, including
      # without limitation the rights to use, copy, modify, merge, publish,
      # distribute, sublicense, and/or sell copies of the Software, and to
      # permit persons to whom the Software is furnished to do so, subject to
      # the following conditions:

      # The above copyright notice and this permission notice shall be
      # included in all copies or substantial portions of the Software.

      # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
      # EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
      # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
      # NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
      # LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
      # OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
      # WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

      def in_groups array, number, fill_with = nil
        # size.div number gives minor group size;
        # size % number gives how many objects need extra accommodation;
        # each group hold either division or division + 1 items.
        division = array.size.div number
        modulo = array.size % number

        # create a new array avoiding dup
        groups = []
        start = 0

        number.times do |index|
          length = division + (modulo > 0 && modulo > index ? 1 : 0)
          groups << last_group = array.slice(start, length)
          last_group << fill_with if fill_with != false &&
            modulo > 0 && length == division
          start += length
        end

        if block_given?
          groups.each{ |g| yield(g) }
        else
          groups
        end
      end
    end
  end
end
