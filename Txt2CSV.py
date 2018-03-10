import csv

'''
测试代码
''' 
def csv_writer(data, filename):
    with open(filename, "w") as csv_file:
        writer = csv.writer(csv_file)
        for line in data:
            writer.writerow(line)
 
if __name__ == "__main__":
    data = []
    with open("S:\\MIT相关文件\\MIT数据采样点\\师妹\\annotations03665.txt") as f:
        for line in f:
            data.append(line.strip().split())
    filename = "11211.csv"
    csv_writer(data, filename)