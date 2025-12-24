#!/bin/bash
set -e

# Configuration
PYTHON_VERSION="3.14"
LAYER_NAME="requests-python314-layer"
REGION="eu-west-3"

echo "🚀 Starting build for Python $PYTHON_VERSION Layer..."

# 1. Clean and create directory structure
rm -rf layer_output
mkdir -p layer_output/python/lib/python$PYTHON_VERSION/site-packages

# 2. Use official AWS Lambda image to install dependencies
# This ensures C-extensions are compiled for Amazon Linux
docker run --rm \
    -v "$PWD/layer_output:/var/task" \
    public.ecr.aws/lambda/python:$PYTHON_VERSION \
    pip install requests -t /var/task/python/lib/python$PYTHON_VERSION/site-packages

# 3. Package the layer
cd layer_output
zip -r ../$LAYER_NAME.zip python
cd ..

echo "✅ Layer created: $LAYER_NAME.zip"

# 4. Optional: Upload to AWS
# Un-comment the line below if you have AWS CLI configured
aws lambda publish-layer-version --layer-name $LAYER_NAME \
    --description "Requests library for Python 3.14" \
    --zip-file fileb://$LAYER_NAME.zip \
    --compatible-runtimes python$PYTHON_VERSION \
    --region $REGION