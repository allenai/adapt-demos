# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from huggingface_hub import HfApi

def extract_conversations(file_path: Path) -> List[Dict]:
    """
    Extract conversations from a JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    conversations = []
    
    def process_conversation(conversation, model_name):
        if conversation:
            messages = []
            for turn in conversation:
                if len(turn) == 2 and turn[0] is not None and turn[1] is not None:
                    messages.append({"role": "user", "content": turn[0]})
                    messages.append({"role": "assistant", "content": turn[1]})
                else:
                    # If any turn is None or not in the expected format, skip this conversation
                    return
            
            if messages:
                conversations.append({
                    "prompt": messages[0]["content"],
                    "messages": messages,
                    "model_name": model_name,
                    "timestamp": data.get("timestamp")
                })
    
    process_conversation(data.get('conversation', []), data.get('model_name'))
    # process_conversation(data.get('conversation_2', []), data.get('model_name_2'))
    
    return conversations

def archive_and_upload_data(
    source_dir: str,
    hf_dataset: str,
    hf_token: str
) -> None:
    """
    Archive JSON files from the source directory, collate conversations, and upload to Hugging Face.

    Args:
    source_dir (str): The directory containing JSON files to be archived.
    hf_dataset (str): The Hugging Face dataset to upload to.
    hf_token (str): The Hugging Face API token for authentication.

    Returns:
    None
    """
    # Create timestamp for the archive folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(source_dir) / f"archive_{timestamp}"
    archive_dir.mkdir(exist_ok=True)

    # Move JSON files to the archive folder and extract conversations
    json_files = list(Path(source_dir).glob("*.json"))
    all_conversations = []
    for file in json_files:
        # skip debug prompts
        if "_debug" in file.name:
            continue
        shutil.move(str(file), str(archive_dir / file.name))
        all_conversations.extend(extract_conversations(archive_dir / file.name))

    print(f"Archived {len(json_files)} files to {archive_dir}")
    print(f"Extracted {len(all_conversations)} conversations")

    # Collate conversations into a single JSON file
    collated_file = archive_dir / f"collated_conversations_{timestamp}.json"
    with open(collated_file, 'w') as f:
        json.dump(all_conversations, f, indent=2)

    print(f"Collated conversations into {collated_file}")

    # Upload the collated file to Hugging Face
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(collated_file),
        path_in_repo=f"raw_data/collated_conversations_{timestamp}.json",
        repo_id=hf_dataset,
        repo_type="dataset",
        token=hf_token
    )
    
    print(f"Uploaded collated conversations to {hf_dataset}/raw_data/collated_conversations_{timestamp}.json")

def main():
    parser = argparse.ArgumentParser(description="Archive user data, collate conversations, and upload to Hugging Face.")
    parser.add_argument("--source_dir", type=str, default="user_data", help="Source directory containing JSON files")
    parser.add_argument("--hf_dataset", type=str, default="allenai/internal-demo-prompts", help="Hugging Face dataset to upload to")
    parser.add_argument("--hf_token", type=str, help="Hugging Face API token")

    args = parser.parse_args()

    archive_and_upload_data(
        source_dir=args.source_dir,
        hf_dataset=args.hf_dataset,
        hf_token=args.hf_token
    )

if __name__ == "__main__":
    main()