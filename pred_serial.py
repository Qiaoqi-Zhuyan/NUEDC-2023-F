import serial
import torch
from model import MLPs

ser = serial.Serial('/dev/ttyACM0', 115200)

string_labels = ['ab,0102', 'ab,0304', 'ab,0506', 'ab,0708', 'ab,0910', 'ab,1112', 'cd,0102', 'cd,0304', 'cd,0506',
                 'cd,0708', 'cd,0910', 'cd,1112', 'ef,0102', 'ef,0304', 'ef,0506', 'ef,0708', 'ef,0910', 'ef,1112',
                 'gh,0102', 'gh,0304', 'gh,0506', 'gh,0708', 'gh,0910', 'gh,1112', 'ij,0102', 'ij,0304', 'ij,0506',
                 'ij,0708', 'ij,0910', 'ij,1112', 'kl,0102', 'kl,0304', 'kl,0506', 'kl,0708', 'kl,0910', 'kl,1112']


def extract_data():
    while True:
        # 读取一个字节
        byte = ser.read(1)

        # 检查是否为帧头的第一个字节
        if byte == b'\xA5':
            # 读取第二个字节
            second_byte = ser.read(1)

            # 检查是否为帧头的第二个字节
            if second_byte == b'\x5A':
                # 读取两个16位无符号整数，先高位后低位
                high_byte1 = ser.read(1)
                low_byte1 = ser.read(1)
                high_byte2 = ser.read(1)
                low_byte2 = ser.read(1)
                high_byte3 = ser.read(1)
                low_byte3 = ser.read(1)

                # 将字节转换为整数
                integer1 = (high_byte1[0] << 8) | low_byte1[0]
                integer2 = (high_byte2[0] << 8) | low_byte2[0]
                integer3 = (high_byte3[0] << 8) | low_byte3[0]

                # 返回提取的三个整数
                return 1, integer1, integer2, integer3


flag = 0

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load("mlp_without_dropout.pt")

    while 1:
        print("wait for data...")
        # 从单片机提取数据
        flag, data1, data2, data3 = extract_data()
        print(f'receive data1: {data1}, receive data2: {data2}, receive data3: {data3}')
        # 过滤数据
        if flag and data1 < 1000 and data2 < 1000 and data3 < 1000:
            input_data = torch.tensor([float(data1), float(data2), float(data3)])
            print(f'input: {input_data}')
            input_data = input_data.to(device)
            # 模型推理
            output = model(input_data)
            value, predicted = torch.max(output.data, 0)
            flag = 0

            # 结果可视化
            data_bytes = bytearray()
            pred = predicted.item()
            class_name = string_labels[pred]
            row = int(pred // 6)
            col = int(pred % 6)
            print(f'pred: {pred}, class-name: {class_name}')

            # 写回单片机
            header = bytearray.fromhex('A55A')
            # first_header = b'\xA5'
            # second_header =  b'\x5A'
            # data_bytes.extend(header.to_bytes(2, byteorder='big'))
            data_bytes.extend(row.to_bytes(2, byteorder='big'))
            data_bytes.extend(col.to_bytes(2, byteorder='big'))
            ser.write(header + data_bytes)
            print(f'send data: {data_bytes}')

            print('\n')
