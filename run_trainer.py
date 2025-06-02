import json
from codebrain.model.trainer import train_digital_twin

if __name__ == "__main__":
    with open("sample_config.json") as f:
        config = json.load(f)
    data_path = "sample_data.json"
    output_path = "digital_twin_model.pth"
    train_digital_twin(data_path, output_path, config) 