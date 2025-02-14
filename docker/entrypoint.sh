#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or fallback to 9001
USER_ID=${LOCAL_USER_ID:-9001}
GROUP_ID=${LOCAL_GROUP_ID:-9001}

echo "Starting with UID : $USER_ID"
groupadd -g $GROUP_ID appuser
useradd --shell /bin/bash -u $USER_ID -g $GROUP_ID -o -c "" -m appuser

# Give ownership to the app directory
chown -R appuser:appuser /app
chown -R appuser:appuser /home/appuser

# Check if running in interactive mode or if the command contains "bash"
if [[ -t 0 || "$*" == *bash* ]]; then
    echo "Running in interactive mode..."
    exec gosu appuser "$@"
    exit 0
fi

# If no command is provided, default to `speech-gen-eval`
if [[ $# -eq 0 ]]; then
    set -- speech-gen-eval
else
    set -- speech-gen-eval "$@"
fi

# Execute process as appuser
exec gosu appuser "$@" 