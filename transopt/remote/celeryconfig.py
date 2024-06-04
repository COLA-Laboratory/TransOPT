## Broker settings.
broker_url = 'redis://localhost:6379/0'
broker_connection_retry_on_startup = True

## Using the database to store task state and results.
result_backend = 'redis://localhost:6379/0'

# If enabled the task will report its status as 'started' 
# when the task is executed by a worker.
task_track_started = True