#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <substring>"
  exit 1
fi

SUBSTRING="$1"

# Check if substring is empty
if [ -z "$SUBSTRING" ]; then
  echo "Error: Substring cannot be empty."
  exit 1
fi

# List all screen sessions and filter by substring
screen_list=$(screen -list | grep -i "$SUBSTRING" | awk -F. '{print $1}' | awk '{print $1}')

if [ -z "$screen_list" ]; then
  echo "No screen sessions found with substring '$SUBSTRING'."
  exit 0
fi

echo "Terminating screen sessions containing '$SUBSTRING'..."

# Terminate each matching screen session
for session in $screen_list; do
  echo "Terminating screen session: $session"
  screen -S "$session" -X quit
done

echo "All matching screen sessions have been terminated."
