# CTM-AbsoluteZero Deployment Guide

This guide provides instructions for deploying CTM-AbsoluteZero in production environments and integrating it with DFZ.

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 20GB minimum
- **GPU**: CUDA-compatible GPU recommended for optimal performance
- **OS**: Linux (recommended), macOS, or Windows

## Installation

### 1. Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 3. Install Language Models

Download the required language models:

```bash
# Create model directories
mkdir -p models/proposer models/solver

# Download models (example using Hugging Face CLI)
huggingface-cli download <model-org>/proposer-model --local-dir models/proposer
huggingface-cli download <model-org>/solver-model --local-dir models/solver
```

Alternatively, you can configure the system to use models from the Hugging Face Hub directly.

### 4. Set Up Configuration

```bash
# Copy example configuration
cp configs/production.yaml configs/my-deployment.yaml

# Edit configuration
nano configs/my-deployment.yaml
```

Ensure you update the following configurations:
- Model paths
- Data storage paths
- Logging settings
- DFZ integration settings

## Running in Production

### Standalone Mode

```bash
# Run with production configuration
python -m src.cli --config configs/my-deployment.yaml dfz
```

### Integrating with DFZ

1. Ensure DFZ is installed and configured
2. Update the DFZ path in your configuration
3. Run in DFZ integration mode

```bash
# Run with DFZ integration
python -m src.cli --config configs/my-deployment.yaml dfz --dfz-path /path/to/dfz
```

### Using Docker

A Dockerfile is provided for containerized deployments:

```bash
# Build the Docker image
docker build -t ctm-absolutezero:latest .

# Run the container
docker run -v /path/to/models:/app/models -v /path/to/data:/app/data \
  -p 8000:8000 ctm-absolutezero:latest
```

## Directory Structure

Create the following directory structure for production deployments:

```
ctm-absolutezero/
├── models/
│   ├── proposer/
│   └── solver/
├── data/
│   ├── embeddings/
│   └── tasks/
├── logs/
├── state/
└── configs/
    └── production.yaml
```

## DFZ Integration

### Plugin Setup

1. Create a symlink or copy the DFZ plugin to your DFZ plugins directory:

```bash
ln -s /path/to/ctm-absolutezero/src/integration/dfz.py /path/to/dfz/plugins/ctm_az_adapter.py
```

2. Enable the plugin in your DFZ configuration.

### API Integration

Alternatively, you can use the HTTP API for integration:

1. Run CTM-AbsoluteZero with the API server:

```bash
python -m src.cli --config configs/production.yaml api --port 8000
```

2. Use the API endpoints to interact with CTM-AbsoluteZero from DFZ.

## Monitoring and Maintenance

### Logs

Logs are stored in the `logs` directory by default. Monitor them for errors and performance issues:

```bash
tail -f logs/ctm-az.log
```

### Metrics

Performance metrics are collected and can be accessed via:

```bash
# Display metrics summary
python -m src.cli metrics

# Export metrics to JSON
python -m src.cli metrics --output metrics.json
```

### Backups

Regular backups of the state and data directories are recommended:

```bash
# Example backup script
tar -czf ctm-az-backup-$(date +%Y%m%d).tar.gz state/ data/
```

### Updates

To update the system:

1. Pull the latest changes
2. Stop the running service
3. Back up the current state
4. Update dependencies
5. Start the service

```bash
git pull
pip install -r requirements.txt
python -m src.cli --config configs/production.yaml dfz
```

## Troubleshooting

### Common Issues

#### Model Loading Errors

- Ensure model paths are correctly configured
- Check for corrupted model files
- Verify sufficient memory for loading models

#### Integration Errors

- Verify DFZ path is correct
- Check network connectivity between services
- Ensure compatible API versions

#### Performance Issues

- Monitor memory usage and GPU utilization
- Adjust batch sizes and worker counts
- Enable performance tracing for bottleneck identification

### Getting Help

If you encounter issues:

1. Check the logs for error messages
2. Refer to the documentation
3. Contact the development team

## Security Considerations

- Use a dedicated service account
- Restrict file permissions
- Keep models and code updated
- Use environment variables for sensitive configuration

## Production Checklist

- [ ] Configuration files updated for production
- [ ] Models downloaded and verified
- [ ] Directory structure created
- [ ] Permissions set correctly
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Backup strategy implemented
- [ ] Security measures in place
- [ ] Integration tested with DFZ
- [ ] Performance tuned for production load

## Advanced Configuration

### Load Balancing

For high-load deployments, set up multiple instances behind a load balancer:

```bash
# Run multiple instances on different ports
python -m src.cli --config configs/instance1.yaml api --port 8001
python -m src.cli --config configs/instance2.yaml api --port 8002
```

### Memory Optimization

Adjust memory usage in the configuration:

```yaml
performance:
  batch_size: 8  # Reduce for lower memory usage
  fp16: true     # Use half-precision for GPU acceleration
  memory_limit: 4096  # MB
```

### Custom Plugins

Extend functionality with custom plugins:

1. Create a plugin class in `src/plugins/`
2. Register the plugin in your configuration
3. Import and use the plugin in your application