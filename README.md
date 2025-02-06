# Chatbot (OpenVINO and Chainlit)

This is a chatbot powered by **OpenVINO**, running a DeepSeek-R1-Distill-Qwen-1.5B model. It uses **Chainlit** for conversational interactions and supports memory retention across exchanges.

---

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

    When prompted, enter the path of the model repo: **hsuwill000/DeepSeek-R1-Distill-Qwen-1.5B-openvino**

    The model files will be saved to `models/DeepSeek-R1-Distill-Qwen-1.5B-openvino/1`

2. Build and run the service.

    ```bash
    docker-compose build
    docker-compose up -d
    ```

    This will:
    - build the chatbot container
    - expose the API on **port 3000**
    - restart on failures automatically

## Usage

1. Access the chatbot: `http://localhost:3000`


2. When you're done, stop and clean-up.

    ```bash
    docker-compose down
    ```

## Configuration

### Customize environment variables
The following are the environment variables you can customize in `app.py`.

| Variable           | Default Value | Description |
|--------------------|--------------|-------------|
| `MODEL_PATH`      | `/app/models/DeepSeek-R1-Distill-Qwen-1.5B-openvino/1` | Path to OpenVINO model |
| `DEFAULT_TEMPERATURE` | `0.7` | Controls response randomness |
| `MAX_LENGTH`      | `4096` | Max input size |
| `INFERENCE_THREADS` | `12` | OpenVINO inference threads |
| `MAX_WORKERS`     | `8` | Parallel request handling |
| `MEMORY_SIZE`     | `3` | Number of previous exchanges to remember |

### Change the model

1. Download a new model from HuggingFace.

    ```bash
    python3 download_model.py
    ```

    When prompted, enter the path of the model repo.

    For example, if the model repo URL is `https://huggingface.co/HelloSun/Qwen2.5-3B-Instruct-openvino`, then enter the path `HelloSun/Qwen2.5-3B-Instruct-openvino`.

2. Update the model in `app.py`, `Dockerfile`, and `docker-compose.yml`. 

## Troubleshooting

**Model Not Found**  
- Ensure the model files are inside `models/` directory.
- Check the model path in `app.py`, `Dockerfile`, and `docker-compose.yml`.

**High CPU Usage**  
- Reduce `INFERENCE_THREADS` and `MAX_WORKERS`.

