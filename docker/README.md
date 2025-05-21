# Docker Deployment for CTM-AbsoluteZero

This directory contains Docker configuration files for deploying the CTM-AbsoluteZero system.

## Components

The deployment consists of the following components:

1. **ctm-az-core**: The main CTM-AbsoluteZero service that handles task generation and execution.
2. **ctm-az-monitor**: A monitoring dashboard for system metrics and performance.
3. **ctm-az-test-runner**: A test runner for automated testing (enabled with the `testing` profile).

## Configuration

The deployment is configured through environment variables:

- `CLAUDE_API_KEY`: API key for Claude models
- `DEEPSEEK_API_KEY`: API key for DeepSeek models
- `TOKEN_OPTIMIZATION_MODE`: Mode for token optimization (aggressive, balanced, minimal)
- `MAX_WORKERS`: Maximum number of concurrent workers for task execution
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `CUDA_ENABLED`: Whether to enable CUDA for GPU acceleration
- `DASH_DEBUG`: Whether to enable debug mode for the Dash monitoring app

## Deployment

### Prerequisites

- Docker and Docker Compose installed
- API keys for Claude and DeepSeek models

### Steps

1. Create a `.env` file in the project root with the required environment variables:

```
CLAUDE_API_KEY=your_claude_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
TOKEN_OPTIMIZATION_MODE=balanced
MAX_WORKERS=4
LOG_LEVEL=INFO
CUDA_ENABLED=true
```

2. Start the services:

```bash
docker-compose up -d
```

3. To include the test runner:

```bash
docker-compose --profile testing up -d
```

4. To stop the services:

```bash
docker-compose down
```

## Access

- Core API: http://localhost:8000
- Monitoring Dashboard: http://localhost:8080

## Health Checks

All services include health check endpoints:

- Core: http://localhost:8000/health
- Monitor: http://localhost:8080/health

## Volumes

The deployment uses the following volumes:

- `./configs`: Configuration files
- `./data`: Application data
- `./models`: Model files
- `./logs`: Log files
- `./state`: State persistence
- `./tests`: Test files
- `./results`: Test results

## Resource Limits

The services have the following resource limits:

- Core: 8GB memory limit, 4GB reservation
- Monitor: 2GB memory limit, 1GB reservation

## Customization

You can customize the deployment by modifying the `docker-compose.yml` file or creating additional configuration files in the `configs` directory.

## Troubleshooting

If you encounter issues with the deployment, check the logs:

```bash
docker-compose logs -f
```

For specific services:

```bash
docker-compose logs -f ctm-az-core
docker-compose logs -f ctm-az-monitor
```