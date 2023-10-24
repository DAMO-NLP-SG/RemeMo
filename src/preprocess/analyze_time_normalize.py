import json, random, os

SAMPLE_SIZE = 100

def main():
    sample_file_path = f"./sample_{SAMPLE_SIZE}.enwiki-20221101_temporal-sentences_special-token-prefix.json"
    original_file_path = "../../../data/pretrain_t5/enwiki-20221101_temporal-sentences_special-token-prefix.json"
    if os.path.exists(sample_file_path):
        print(f"Sample file already exists at {sample_file_path}")
        with open(sample_file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines() if line.strip() != '']
    else:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines() if line.strip() != '']
        random.shuffle(data)
        data = data[:SAMPLE_SIZE]
        with open(sample_file_path, 'w', encoding='utf-8') as f:
            f.writelines([json.dumps(d) + '\n' for d in data])
    annotated_data_path = f"./sample_{SAMPLE_SIZE}.annotated.enwiki-20221101_temporal-sentences_special-token-prefix.json"
    if os.path.exists(annotated_data_path):
        print(f"Annotated data file already exists at {annotated_data_path}")
        with open(annotated_data_path, 'r', encoding='utf-8') as f:
            annotated_data = json.load(f)
    else:
        annotated_data = [[] for _ in range(SAMPLE_SIZE)]
    # annotated_data = [[]*SAMPLE_SIZE]
    start_idx = 0
    for i, d in enumerate(annotated_data):
        if d == []:
            if i > 0:
                start_idx = i - 1
            break
    try:
        for i, d in enumerate(data):
            if i >= start_idx:
                while True:
                    x = input(
                        '******\nNo.{}: Enter a normalized time expression (e.g., "[20100101, 20120304]") for the following sentence:\n\n{}\n\nAlready input: {} \n\n(Input "q" to quit the program, "s" to skip the current one.):\n'.format(i+1, d['text'], annotated_data[i])
                    )
                    if x == 'q':
                        raise KeyboardInterrupt
                    elif x == 's':
                        break
                    else:
                        try:
                            x = json.loads(x)
                            annotated_data[i].append(x)
                        except:
                            print('Invalid input. Please try again.\n')
                    print("\n")
    except:
        with open(annotated_data_path, 'w', encoding='utf-8') as f:
            json.dump(annotated_data, f)
        raise KeyboardInterrupt

if __name__ == '__main__':
    main()