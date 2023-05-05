from pathlib import Path
import json

domains = ['Video_Games', 'Toys_and_Games', 'Tools_and_Home_Improvement', 'Sports_and_Outdoors', 'Pet_Supplies', 'Patio_Lawn_and_Garden', 'Office_Products', 'Musical_Instruments', 'Movies_and_TV', 'Kindle_Store']

datasets = ['/data3/zl/lifelong/dat/dsc/' + domain for domain in domains]

if __name__ == "__main__":
    f_name = "dsc_random_10"
    ntasks = 10
    idrandom = 0

    with open(f_name, 'r') as f_random_seq:
        random_seq = f_random_seq.readlines()[idrandom].split()

    for domian in random_seq:
        file_path = Path('/data3/zl/lifelong/dat/dsc/') / Path(domian) / Path('test.json')

        with open(file_path, 'r') as f:
            data = json.load(f)

        with open(Path('test_json') / Path(domian + '.json'), 'w') as h:
            json.dump(data, h)