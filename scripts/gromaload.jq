#!/usr/bin/env -S jq -rf
.files_by_relevance
| map("/add " + .[1])        # prepend “/add ” to each filename
| join("\n")                 # newline-separate them
