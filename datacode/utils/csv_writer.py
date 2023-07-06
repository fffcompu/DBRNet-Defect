import csv

def csv_writer(dict,path):
    a = []
    for headers in sorted(dict.keys()):  # 把字典的键取出来
        a.append(headers)
    header = a  # 把列名给提取出来，用列表形式呈现
    with open('{}.csv'.format(path), 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        writer.writeheader()  # 写入列名
        writer.writerows([dict])  # 写入数据