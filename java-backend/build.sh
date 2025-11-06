#!/bin/bash

# Build the application
echo "Building Spring Boot application..."
mvn clean package -DskipTests

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
else
    echo "Build failed!"
    exit 1
fi
