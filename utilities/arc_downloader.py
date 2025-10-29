import requests
import zipfile
import os
import io
import re
import json


zip_url = 'https://codeload.github.com/fchollet/ARC-AGI/zip/refs/heads/master'
subset_names = ['training', 'evaluation']


def download_arc_data(arc_data_path):
    # check if files are already there
    required_files = []
    for subset in subset_names:
        required_files.append(os.path.join(arc_data_path, f'arc-agi_{subset}_challenges.json'))
        required_files.append(os.path.join(arc_data_path, f'arc-agi_{subset}_solutions.json'))
    if all(map(os.path.isfile, required_files)): return


    # download repo
    r = requests.get(zip_url)
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # extract subsets
    extract_id = re.compile('^ARC-AGI-master/data/([a-z]+)/([a-z0-9]+)[.]json')
    datasets = {}
    for f in z.filelist:
        id = extract_id.match(f.filename)
        if id:
            if id.group(1) not in datasets: datasets[id.group(1)] = {}
            datasets[id.group(1)][id.group(2)] = json.loads(z.read(f))

    # store challenges and solutions seperately
    os.makedirs(arc_data_path, exist_ok=True)
    for subset, challenges in datasets.items():
        solutions = {}
        for k, v in challenges.items():
            assert v.pop('name', k) == k  # remove name tags that occur inconsistently in the data
            solutions[k] = [t.pop('output') for t in v['test']]
        with open(os.path.join(arc_data_path, f'arc-agi_{subset}_challenges.json'), 'w') as f: json.dump(challenges, f)
        with open(os.path.join(arc_data_path, f'arc-agi_{subset}_solutions.json'), 'w') as f: json.dump(solutions, f)
        print(f'Downloaded arc {subset} set.')
