#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <python_script> <num_runs> <concurrent_runs> <screen_name_template>"
  exit 1
fi

PYTHON_SCRIPT="$1"
NUM_RUNS="$2"
CONCURRENT_RUNS="$3"
SCREEN_NAME_TEMPLATE="$4"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script '$PYTHON_SCRIPT' not found!"
  exit 1
fi

# Check if NUM_RUNS and CONCURRENT_RUNS are positive integers
if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || ! [[ "$CONCURRENT_RUNS" =~ ^[0-9]+$ ]]; then
  echo "Error: NUM_RUNS and CONCURRENT_RUNS must be positive integers."
  exit 1
fi

# Check if CONCURRENT_RUNS is greater than 0
if [ "$CONCURRENT_RUNS" -le 0 ]; then
  echo "Error: CONCURRENT_RUNS must be greater than 0."
  exit 1
fi

# Function to start a screen session with the Python script
start_screen_session() {
  local index=$1
  local screen_name=$(printf "$SCREEN_NAME_TEMPLATE" "$index")
  screen -dmS "$screen_name" bash -c "python3 '$PYTHON_SCRIPT'; exit"
  echo "Started screen session '$screen_name' with command: python3 '$PYTHON_SCRIPT'"
}

# Function to count the number of running screen sessions matching the pattern
count_running_screens() {
  local pattern=$(echo "$SCREEN_NAME_TEMPLATE" | sed 's/%d/.*.*/') # Replace %d with a regex pattern
  screen -list | grep -E "$pattern" | grep -c 'Detached'
}

# Loop to start the screen sessions
for ((i = 1; i <= NUM_RUNS; i++)); do
  start_screen_session "$i"

  # Wait if we have reached the maximum number of concurrent runs
  while [ "$(count_running_screens)" -ge "$CONCURRENT_RUNS" ]; do
    echo "Waiting for concurrent runs to finish..."
    sleep 5
  done
done

# Wait for all remaining screen sessions to finish
echo "Waiting for all remaining runs to finish..."
while [ "$(count_running_screens)" -gt 0 ]; do
  sleep 5
done

echo "All runs are completed."
