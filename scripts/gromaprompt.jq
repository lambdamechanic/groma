#!/usr/bin/env -S jq -rf
"The relevant data may be in these files:\n"
+ (.files_by_relevance
  | map("- " + .[1])      # pick each filename and add a bullet
  | join("\n")            # put one per line
  )
