#!/bin/bash

# Define SESSION_NAME
SESSION_NAME="experiment_server"

activate_conda_env() {
    local env_name="$1"
    local target_pane="$2"
    
    if [ -n "$env_name" ]; then
        tmux send-keys -t "$target_pane" "conda activate $env_name" C-m
    fi
}

display_shortcuts() {
    local target_pane="$1"
    
    # ANSI escape codes for bold and colored text
    local BOLD="\033[1m"
    local COLOR_RED="\033[31m"
    local RESET="\033[0m"
    
    # Set display-time to 10 seconds (10000 milliseconds)
    tmux set-option -t "$target_pane" display-time 10000
    
    # Display the shortcuts using tmux's display-message
    tmux display-message -t "$target_pane" "${BOLD}${COLOR_RED}SHORTCUTS: Ctrl-b n (next window), Ctrl-b p (previous window), Ctrl-b d (detach)${RESET}"
}

run_experiment_server() {
    local env_name="$1"
    
    # Start a new tmux session
    tmux new-session -d -s "$SESSION_NAME"
    
    # Activate conda environment (if specified) and run the Celery worker in the first window
    display_shortcuts "$SESSION_NAME:0"
    activate_conda_env "$env_name" "$SESSION_NAME:0"
    tmux send-keys -t "$SESSION_NAME:0" 'celery -A experiment_tasks.celery worker --loglevel=info' C-m
    
    # Run Flask in a new window
    tmux new-window -t "$SESSION_NAME:1"
    display_shortcuts "$SESSION_NAME:1"
    activate_conda_env "$env_name" "$SESSION_NAME:1"
    tmux send-keys -t "$SESSION_NAME:1" 'python experiment_server.py' C-m
    
    # Attach to the 'experiment_server' session
    tmux attach -t "$SESSION_NAME"
}

case "$1" in
    start)
        # Get the currently activated conda environment
        CURRENT_CONDA_ENV=$(conda env list | grep '*' | awk '{print $1}')
        
        if [ -z "$CURRENT_CONDA_ENV" ]; then
            echo "No Conda environment is currently activated."
            run_experiment_server ""
        else
            run_experiment_server "$CURRENT_CONDA_ENV"
        fi
    ;;
    
    attach)
        tmux attach -t "$SESSION_NAME"
    ;;
    
    stop)
        tmux kill-session -t "$SESSION_NAME"
    ;;
    
    *)
        echo "Usage: $0 {start|attach|stop}"
        exit 1
    ;;
esac