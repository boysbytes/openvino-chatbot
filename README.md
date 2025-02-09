# Chatbot (OpenVINO and Chainlit)

This is a chatbot powered by [**OpenVINO**](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html), running a DeepSeek-R1-Distill-Qwen-1.5B model. It uses [**Chainlit**](https://docs.chainlit.io/get-started/overview) for conversational interactions and supports memory retention across exchanges.

## Key Features

- **Runs with OpenVINO**.
- **Conversational Memory** – remembers the last few interactions.
- **Configurable Settings** – adjust model parameters easily.
- **Dockerized** – deploy seamlessly with Docker & Docker Compose.


## Installation

1. Clone the repository.

    ```bash
    git clone https://github.com/boysbytes/openvino-chatbot.git
    cd openvino-chatbot
    ```

2. Download the model.

    ```bash
    python3 download_model.py
    ```

    When prompted, enter the path of the model repo, for example, `https://huggingface.co/boysbytes/DeepSeek-R1-Distill-Qwen-1.5B-openvino-8bit`

    The model files will be saved to `models/DeepSeek-R1-Distill-Qwen-1.5B-openvino-8bit/1`

2. Build and run the service.

    ```bash
    docker-compose build
    docker-compose up -d
    ```

## Usage

1. Access the chatbot: `http://localhost:3000`


2. When you're done, stop and clean-up.

    ```bash
    docker-compose down
    ```

## Configuration

### Customize environment variables
The following are the environment variables you can customize in `.env`.

| Variable           | Default Value | Description |
|--------------------|--------------|-------------|
| `MODEL_PATH`      | `/app/models/DeepSeek-R1-Distill-Qwen-1.5B-openvino-8bit/1` | Path to OpenVINO model |
| `DEFAULT_TEMPERATURE` | `0.7` | Controls response randomness |
| `MAX_LENGTH`      | `8192` | Max input size |
| `INFERENCE_THREADS` | `12` | OpenVINO inference threads |
| `MAX_WORKERS`     | `8` | Parallel request handling |
| `MEMORY_SIZE`     | `3` | Number of previous exchanges to remember |
| `MAX_NEW_TOKENS`     | `4096` | Max response size |

### Change the model

1. Download a new model from HuggingFace.

    ```bash
    python3 download_model.py
    ```

    When prompted, enter the path of the model repo, for example, `https://huggingface.co/boysbytes/DeepSeek-R1-Distill-Qwen-1.5B-openvino-8bit`.

2. Update the model path in `.env`. 

## Troubleshooting

**Model Not Found**  
- Ensure the model files are inside `models/` directory.
- Check the model path in `.env`.

**High CPU Usage**  
- Reduce `INFERENCE_THREADS` and `MAX_WORKERS`.

