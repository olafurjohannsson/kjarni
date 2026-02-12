#!/bin/bash

set -e

echo "Downloading Simple Wikipedia dataset..."
python3 -c "
from datasets import load_dataset
import os
ds = load_dataset('wikimedia/wikipedia', '20231101.simple', split='train[:10000]', trust_remote_code=True)
os.makedirs('simple_wiki', exist_ok=True)
for i, item in enumerate(ds):
    if len(item['text']) < 200: continue
    safe = ''.join(c if c.isalnum() or c == '_' else '_' for c in item['title'][:50])
    open(f'simple_wiki/{safe}_{i}.txt', 'w').write(f\"# {item['title']}\n\n{item['text']}\")
print(f'Created {len(os.listdir(\"simple_wiki\"))} documents')
"