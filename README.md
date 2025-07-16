# LLM Fine-tuning Project for Character Role-play

This repository contains a comprehensive workflow for fine-tuning large language models (LLMs) to simulate specific characters in a role-playing context, with a complete pipeline from data collection to deployment.

## Project Overview

This project implements an end-to-end solution for:

1. **Data Collection & Processing**: Web scraping for character dialogues and descriptions, formatting conversational data for training.
2. **Model Fine-tuning**: Scripts for fine-tuning various language models (Llama 3 8B, Qwen2 7B) with optimizations like LoRA and DeepSpeed.
3. **Inference & Deployment**: Discord bot implementation with session management and dynamic prompting.
4. **Feedback Collection**: System for gathering user feedback to improve model performance.

## Key Components

### Data Processing

- `Scraping pipeline starrail.ipynb`: Web scraper for collecting character dialogues and descriptions
- `data/`: Directory containing background information and scraped dialogues
- `event_prompts.csv`: Character prompts with daily schedule variations for dynamic context
- `all_conversations.json`: Formatted conversations for training

### Model Training

- `L3 8B Lunaris training.py`: Main training script for fine-tuning Llama 3 8B model using DeepSpeed
- `Llama3_2_(1B_and_3B)_Conversational.ipynb`: Step-by-step notebook for fine-tuning smaller Llama 3 models
- `Qwen2_5_(7B)_Alpaca.ipynb`: Fine-tuning workflow for Qwen2 7B model using Unsloth
- `ds_config.json`: DeepSpeed configuration for distributed training

### Inference & Deployment

- `discord_simulated_universe_trial_1.py`: Discord bot for model deployment with:
  - Session management
  - Character context tracking
  - Time-based prompting
  - User feedback collection
- `inference_docker_test.py`: Test script for containerized inference

### Advanced Training Techniques

- `unsloth_compiled_cache/`: Compiled training modules for advanced techniques:
  - `UnslothSFTTrainer.py`: Supervised fine-tuning
  - `UnslothDPOTrainer.py`: Direct Preference Optimization
  - `UnslothPPOTrainer.py`: Proximal Policy Optimization
  - Other RLHF implementations

## Technical Implementation

### Fine-tuning Approach

The project utilizes parameter-efficient fine-tuning with LoRA adapters, allowing customization of large models with minimal computational resources. Key technical aspects include:

- **DeepSpeed Integration**: ZeRO optimization for memory efficiency
- **4-bit Quantization**: Model compression for faster inference
- **Unsloth Acceleration**: 2x faster training and inference
- **Custom Loss Functions**: Focused training on assistant responses

### Discord Bot Architecture

The bot implements:
- **Dynamic Context Management**: Tracking user sessions and conversation history
- **Time-aware Prompting**: Adjusting character contexts based on time
- **Feedback Collection**: Capturing user preferences for model improvement
- **Firebase Integration**: Storing conversation logs and user data

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU for training
- Discord account for bot deployment

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/llm-fine-tuning-project.git
cd llm-fine-tuning-project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Environment setup
```bash
# Create a .env file with your Discord token
echo "DISCORD_BOT_TOKEN=your_token_here" > .env
```

### Training a Model

Run the main training script:
```bash
python "L3 8B Lunaris training.py"
```

Or use one of the notebooks for a step-by-step approach:
```bash
jupyter notebook "Llama3_2_(1B_and_3B)_Conversational.ipynb"
```

### Deploying the Discord Bot

1. Ensure your model is saved in the correct location
2. Run the Discord bot:
```bash
python discord_simulated_universe_trial_1.py
```

## Project Structure

```
fine-tune/
├── data/                           # Training data
│   ├── background.json             # Character background information
│   └── final_cleaned_scraped_dialogue_and_descriptions.txt  # Processed dialogues
├── lunaris_finetuned/              # Fine-tuned model outputs
├── results/                        # Training results and logs
├── unsloth_compiled_cache/         # Advanced training modules
├── L3 8B Lunaris training.py       # Main training script
├── discord_simulated_universe_trial_1.py  # Discord bot for deployment
├── ds_config.json                  # DeepSpeed configuration
├── event_prompts.csv               # Dynamic character prompts
├── all_conversations.json          # Formatted training conversations
├── Llama3_2_(1B_and_3B)_Conversational.ipynb  # Llama 3 fine-tuning notebook
├── Qwen2_5_(7B)_Alpaca.ipynb       # Qwen fine-tuning notebook
└── Scraping pipeline starrail.ipynb  # Data collection workflow
```

## Future Work

- Implement RLHF pipeline with human feedback
- Add multi-character interaction capabilities
- Optimize for mobile deployment with smaller models
- Expand data collection to more characters and universes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Unsloth team for fine-tuning optimizations
- DeepSpeed for distributed training capabilities
- Hugging Face for transformer models and libraries