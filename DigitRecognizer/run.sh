#!/bin/bash

# Create necessary directories
mkdir -p models data/raw data/processed data/feedback

# Run the application
python main.py "$@" 