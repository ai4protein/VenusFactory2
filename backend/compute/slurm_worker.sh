#!/bin/sh
# The controller supplies a frozen Python entrypoint, never shell text.
exec "$@"
