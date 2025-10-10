import itertools
import json
import random


def load_data(file_path):
    data = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def augment_data(nuggets_list, batch_size_range=(1, 10)):
    augmented_samples = []
    grouped_nuggets = {
        'support': [n for n in nuggets_list if n['match'] == 'support'],
        'partial_support': [n for n in nuggets_list if n['match'] == 'partial_support'],
        'not_support': [n for n in nuggets_list if n['match'] == 'not_support']
    }

    match_types = ['support', 'partial_support', 'not_support']

    for batch_size in range(batch_size_range[0], batch_size_range[1] + 1):
        possible_counts = []
        for p in itertools.product(range(batch_size + 1), repeat=len(match_types)):
            if sum(p) == batch_size:
                possible_counts.append(p)

        for counts in possible_counts:
            if all(counts[i] <= len(grouped_nuggets[match_types[i]]) for i in range(len(match_types))):
                for permutation in itertools.permutations(range(len(match_types))):
                    new_nuggets = []
                    for i in permutation:
                        match_type = match_types[i]
                        count = counts[i]
                        if count > 0:
                            drawn_nuggets = random.sample(grouped_nuggets[match_type], count)
                            new_nuggets.extend(drawn_nuggets)
                    if new_nuggets not in [s['nuggets_list'] for s in augmented_samples]:
                        random.shuffle(new_nuggets)
                        augmented_samples.append({'nuggets_list': new_nuggets})

    return augmented_samples

def main():
    input_file = '/path/to/your/input.jsonl'
    final_output_file = '/path/to/your/input_aug.jsonl'
    original_data = load_data(input_file)
    augmented_only_data = []
    for item in original_data:
        new_samples = augment_data(item['nuggets_list'])
        for sample in new_samples:
            augmented_only_data.append({
                'qid': item['qid'],
                'query': item['query'],
                'block_text': item['block_text'],
                'nuggets_list': sample['nuggets_list']
            })
    
    print(f"共生成 {len(augmented_only_data)} 条增广样本")
    downsampled_augmented_data = []
    downsampled_count = 0
    for item in augmented_only_data:
        nuggets = item['nuggets_list']
        total_nuggets = len(nuggets)
        if total_nuggets > 5:
            not_support_nuggets = [n for n in nuggets if n['match'] == 'not_support']
            num_not_support = len(not_support_nuggets)
            if num_not_support / total_nuggets > 0.5:
                downsampled_count += 1
                num_to_remove = int(num_not_support * 0.2)
                nuggets_to_keep = [n for n in nuggets if n['match'] != 'not_support']
                nuggets_to_remove = random.sample(not_support_nuggets, num_to_remove)
                remaining_not_support = [n for n in not_support_nuggets if n not in nuggets_to_remove]
                new_nuggets_list = nuggets_to_keep + remaining_not_support
                new_item = {
                    'qid': item['qid'],
                    'query': item['query'],
                    'block_text': item['block_text'],
                    'nuggets_list': new_nuggets_list
                }
                downsampled_augmented_data.append(new_item)
            else:
                downsampled_augmented_data.append(item)
        else:
            downsampled_augmented_data.append(item)

    sample_size = int(len(downsampled_augmented_data) * 0.1)
    if sample_size > 0:
        sampled_augmented_data = random.sample(downsampled_augmented_data, sample_size)
    else:
        sampled_augmented_data = []
    final_dataset = original_data + sampled_augmented_data
    print(f"最终数据集共包含 {len(final_dataset)} 条样本")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()