import pandas as pd
import re

def analyse_sensitivity_test_log(log_path, start_line, end_line) -> dict:
    data = {
        'Layer Pruned': [],
        'Block Num': [],
        'Sparsity': [],
        'Acc.': []
    }
    
    f = open(log_path, 'r', encoding='utf-8')
    count = 0
    while True:
        line = f.readline()
        count += 1
        if count <= start_line:
            continue
        if count > end_line:
            break
        if line == None:
            break
        if line.startswith('Layer pruned'):
            # extract cfg
            matches = re.findall(r"\s*([\d.]+)", line)
            layer, block_num, sp = matches[0], matches[1], matches[2]
            # extract acc
            acc_line = f.readline()
            assert acc_line.startswith('Before pruning'), "did not catch an accuracy report"
            matches = re.findall(r"After pruning: acc=([\d.]+)", acc_line)
            acc = matches[0]
            # append data
            data['Layer Pruned'].append(layer)
            data['Block Num'].append(block_num)
            data['Sparsity'].append(sp)
            data['Acc.'].append(acc)
            # print(f'Layer={layer}, block_num={block_num}, sp={sp}, acc.={acc}')
    f.close()
    
    print(f'Finished extracting {count} pieces of trial data.')
    df = pd.DataFrame(data)
    xlsx_path = log_path.rstrip('.log') + '.xlsx'
    df.to_excel(xlsx_path, index=False)
        
if __name__ == '__main__':
    analyse_sensitivity_test_log('./logs/sensitivity_test_0817_1836.log', start_line=112, end_line=100000)