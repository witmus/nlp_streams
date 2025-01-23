f = open('dataset.csv', 'rb')
f_output = open('dataset_converted.csv', 'ab')


while True:
    b = f.read(1)
    if not b:
        f.close()
        f_output.close()
        break

    if int.from_bytes(b) <= 0x80:
        f_output.write(b)
        
print('finish')
