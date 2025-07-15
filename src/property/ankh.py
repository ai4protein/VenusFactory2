import sys
import os

sys.path.append(os.getcwd())

import argparse
import torch
import json
import warnings
import csv
from pathlib import Path
from tqdm import tqdm
from transformers import T5EncoderModel, AutoTokenizer
from typing import List, Tuple, Optional
# Assuming your AdapterModel is defined at this path
from src.models.adapter_model import AdapterModel


def parse_fasta(file_path: str) -> List[Tuple[str, str]]:
    """
    A simple FASTA parser that supports multiple sequences.
    Returns a list of tuples, where each element is (header, sequence).
    """
    sequences = []
    header = None
    sequence_parts = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header:
                    sequences.append((header, "".join(sequence_parts)))
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        if header:
            sequences.append((header, "".join(sequence_parts)))
    return sequences


def load_model_and_tokenizer(args):
    """
    Loads the pre-trained PLM, tokenizer, and the trained AdapterModel.
    """
    print("---------- Loading Model and Tokenizer ----------")
    device = torch.device("cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu")

    # Prioritize the adapter path provided by the command line.
    model_path = args.adapter_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Adapter model file not found: {model_path}")

    config_path = os.path.join(os.path.dirname(model_path), "lr5e-4_bt12k_ga8.json")
    model_adapter_path = os.path.join(
        os.path.dirname(model_path), "lr5e-4_bt12k_ga8.pt")
    # Load model configuration from config.json, but command-line arguments have higher priority.
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            for key, value in config.items():
                setattr(args, key, value)
            args.structure_seq = ""
    except FileNotFoundError:
        print(
            f"Model config not found at {config_path}. Using command line arguments only.")


    # Load PLM (Pre-trained Language Model).
    tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large", do_lower_case=False)
    plm_model = T5EncoderModel.from_pretrained("ElnaggarLab/ankh-large")

    # Instantiate AdapterModel and load the trained weights.
    model = AdapterModel(args)
    model.load_state_dict(torch.load(model_adapter_path, map_location=device))
    model.to(device).eval()

    return model, plm_model, tokenizer, device


def predict(model, data_dict, device, args, plm_model):
    """
    Performs inference on a single sample.
    """
    for k, v in data_dict.items():
        data_dict[k] = v.to(device)

    with torch.no_grad():
        outputs = model(plm_model, data_dict)

        if args.problem_type == "regression":
            predictions = outputs.squeeze().item()
            return {"prediction": predictions}

        elif args.problem_type == "single_label_classification":
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            class_probs = probabilities.squeeze().tolist()
            return {
                "predicted_class": predicted_class,
                "probabilities": class_probs
            }

        elif args.problem_type == "multi_label_classification":
            sigmoid_outputs = torch.sigmoid(outputs)
            predictions = (sigmoid_outputs > 0.5).int().squeeze().tolist()
            probabilities = sigmoid_outputs.squeeze().tolist()

            # Ensure the result is a list, even for a single label.
            if not isinstance(predictions, list):
                predictions = [predictions]
            if not isinstance(probabilities, list):
                probabilities = [probabilities]

            return {
                "predictions": predictions,
                "probabilities": probabilities
            }
        else:
            # Default case for unknown problem types
            return {"raw_output": outputs.squeeze().tolist()}


def main():
    parser = argparse.ArgumentParser(description="Protein Prediction Pipeline")
    parser.add_argument("--fasta_file", type=str, required=True,
                        help="Input FASTA file (can contain multiple sequences)")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to the trained AdapterModel file")
    parser.add_argument("--output_csv", type=str, default="prediction_results.csv",
                        help="Path to save the output CSV file")
    args = parser.parse_args()

    # Load the model and tokenizer (only once).
    model, plm_model, tokenizer, device = load_model_and_tokenizer(args)

    # Parse all sequences from the FASTA file.
    sequences_to_predict = parse_fasta(args.fasta_file)
    if not sequences_to_predict:
        print("No sequences found in the FASTA file.")
        return

    all_results = []
    print(f"\nFound {len(sequences_to_predict)} sequences. Starting prediction...")

    # Process each sequence one by one.
    for header, sequence in tqdm(sequences_to_predict, desc="Predicting sequences"):
        # Prepare model input.
        aa_inputs = tokenizer(
            [sequence], return_tensors="pt", padding=True, truncation=True)
        data_dict = {
            "aa_seq_input_ids": aa_inputs["input_ids"],
            "aa_seq_attention_mask": aa_inputs["attention_mask"]
        }

        # Run prediction.
        result_data = predict(model, data_dict, device, args, plm_model)

        # Combine header, sequence, and prediction results using dictionary unpacking.
        output_row = {
            "header": header,
            "sequence": sequence,
            **result_data
        }
        all_results.append(output_row)

    # Write all results to the CSV file.
    output_path = args.output_csv
    print(f"\nWriting results to {output_path}...")

    if all_results:
        # Use the first result to determine the CSV column headers.
        fieldnames = list(all_results[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_data in all_results:
                # Convert list-like probabilities to a JSON string for storage.
                for key, value in row_data.items():
                    if isinstance(value, list):
                        row_data[key] = json.dumps(value)
                writer.writerow(row_data)

    print("\n---------- Prediction Complete ----------")
    print(f"{len(all_results)} predictions saved to {output_path}")


if __name__ == "__main__":
    main()
